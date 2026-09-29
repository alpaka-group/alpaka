/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <alpakaTest/testMacros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <type_traits>

using namespace alpaka;

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

template<uint32_t T_blockSize>
struct SharedBlockIotaKernel
{
    template<typename T>
    ALPAKA_FN_ACC void operator()(T const& acc, auto out, auto numBlocks, auto blockExtents) const
    {
        auto& shared = declareSharedVar<uint32_t[T_blockSize], uniqueId()>(acc);

        for(auto blockIdx : onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{numBlocks}))
        {
            auto blockOffset = blockIdx * blockExtents;
            for(auto inBlockOffset : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{blockExtents}))
            {
                uint32_t id = (T_blockSize - 1u - inBlockOffset).x();
                shared[id] = id;
            }

            alpaka::onAcc::syncBlockThreads(acc);
            for(auto inBlockOffset : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{blockExtents}))
            {
                out[blockOffset + inBlockOffset] = (blockOffset + shared[inBlockOffset.x()]).x();
            }
        }
    }
};

TEMPLATE_LIST_TEST_CASE("block shared iota", "[sharedMem]", TestApis)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);
    onHost::Queue queue = device.makeQueue();

    constexpr Vec numBlocks = Vec{2u};
    constexpr Vec blockExtent = Vec{128u};
    constexpr Vec dataExtent = numBlocks * blockExtent;
    auto dBuff = onHost::alloc<uint32_t>(device, dataExtent);

    auto hBuff = onHost::allocHostLike(dBuff);
    alpaka::onHost::wait(queue);

    queue.enqueue(
        onHost::FrameSpec{numBlocks / 2u, blockExtent, exec},
        KernelBundle{SharedBlockIotaKernel<blockExtent.x()>{}, dBuff, numBlocks, blockExtent});
    alpaka::onHost::memcpy(queue, hBuff, dBuff);
    alpaka::onHost::wait(queue);

    auto* ptr = onHost::data(hBuff);
    for(uint32_t i = 0u; i < dataExtent; ++i)
    {
        CHECK(i == ptr[i]);
    }
}

/** Validate shared memory aliasing and uniqueness.
 *
 * If a id during the shared memory declaration is used twice the same memory should be returned.
 * Different id's should produce independent shared memory.
 */
struct SharedMemAlias
{
    template<typename T>
    ALPAKA_FN_ACC void operator()(T const& acc, auto result) const
    {
        bool test = true;
        auto& s0 = declareSharedVar<uint32_t, uniqueId()>(acc);
        auto& s1 = declareSharedVar<uint32_t, uniqueId()>(acc);
        test = test && &s0 != &s1;

        auto& s2 = declareSharedVar<uint32_t, 42>(acc);
        auto& s3 = declareSharedVar<uint32_t, 42>(acc);
        test = test && &s2 == &s3;

        // check that we not create an alias to the first two variables
        test = test && &s2 != &s0;
        test = test && &s2 != &s1;

        auto a0 = declareSharedMdArray<uint32_t, uniqueId()>(acc, CVec<uint32_t, 2u>{});
        auto a1 = declareSharedMdArray<uint32_t, uniqueId()>(acc, CVec<uint32_t, 2u>{});
        test = test && a0 != a1;

        auto a2 = declareSharedMdArray<uint32_t, 42>(acc, CVec<uint32_t, 2u>{});
        auto a3 = declareSharedMdArray<uint32_t, 42>(acc, CVec<uint32_t, 2u>{});
        test = test && a2 == a3;

        // check that we not create an alias to the first two arrays
        test = test && a2 != a0;
        test = test && a2 != a1;

        // the address of a shared memory array and normal shared variable is not allowed to be equal even if the same
        // id is used.
        test = test && &s2 != &a2[0u];

        result[0] = test;
    }
};

struct DynSharedMemMember
{
    uint32_t dynSharedMemBytes = 32u;

    template<typename T>
    ALPAKA_FN_ACC void operator()(T const& acc, auto result) const
    {
        bool test = true;

        // set dynamic shared memory
        auto* dynS0 = getDynSharedMem<uint32_t>(acc);
        auto* dynS1 = getDynSharedMem<uint32_t>(acc);

        test = test && dynS0 == dynS1;

        result[0] = test;
    }
};

struct DynSharedMemTrait
{
    template<typename T>
    ALPAKA_FN_ACC void operator()(T const& acc, auto result) const
    {
        bool test = true;

        // set dynamic shared memory
        auto* dynS0 = getDynSharedMem<uint32_t>(acc);
        auto* dynS1 = getDynSharedMem<uint32_t>(acc);

        test = test && dynS0 == dynS1;

        result[0] = test;
    }
};

namespace alpaka::onHost::trait
{
    template<concepts::ThreadSpec T_Spec>
    struct BlockDynSharedMemBytes<DynSharedMemTrait, T_Spec>
    {
        BlockDynSharedMemBytes(DynSharedMemTrait const& kernel, T_Spec const& spec)
        {
            alpaka::unused(kernel, spec);
        }

        uint32_t operator()(auto const&... args) const
        {
            alpaka::unused(args...);
            return 32;
        }
    };
} // namespace alpaka::onHost::trait

TEMPLATE_LIST_TEST_CASE("block shared alias", "[SharedMem]", TestApis)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);

    onHost::Queue queue = device.makeQueue();
    constexpr Vec numBlocks = Vec{1u};
    constexpr Vec blockExtent = Vec{1u};

    auto dBuff = onHost::alloc<bool>(device, Vec{1u});

    auto hBuff = onHost::allocHostLike(dBuff);
    alpaka::onHost::wait(queue);
    {
        queue.enqueue(onHost::FrameSpec{numBlocks, blockExtent, exec}, KernelBundle{SharedMemAlias{}, dBuff});
        alpaka::onHost::memcpy(queue, hBuff, dBuff);
        alpaka::onHost::wait(queue);
        CHECK(hBuff[0] == true);
    }
    {
        queue.enqueue(onHost::FrameSpec{numBlocks, blockExtent, exec}, KernelBundle{DynSharedMemMember{}, dBuff});
        alpaka::onHost::memcpy(queue, hBuff, dBuff);
        alpaka::onHost::wait(queue);
        CHECK(hBuff[0] == true);
    }
    {
        queue.enqueue(onHost::FrameSpec{numBlocks, blockExtent, exec}, KernelBundle{DynSharedMemTrait{}, dBuff});
        alpaka::onHost::memcpy(queue, hBuff, dBuff);
        alpaka::onHost::wait(queue);
        CHECK(hBuff[0] == true);
    }
}

template<typename T_Idx>
struct IndexTypeKernel
{
    constexpr void operator()(alpaka::onAcc::concepts::Acc auto const& acc, alpaka::concepts::IMdSpan auto result)
        const
    {
        auto a = declareSharedMdArray<float, uniqueId()>(acc, CVec<T_Idx, 2u>{});
        using SharedMdArrayType = decltype(a);
        result[0u] = std::is_same_v<typename SharedMdArrayType::index_type, T_Idx>;
        result[1u] = std::is_same_v<typename ALPAKA_TYPEOF(a.getExtents())::value_type, T_Idx>;
    }
};

template<typename T_Idx>
void test_index_type(auto& queue, auto const& exec, auto name)
{
    auto dBuff = onHost::alloc<bool>(queue.getDevice(), Vec{2u});
    auto hBuff = onHost::allocHostLike(dBuff);
    onHost::wait(queue);

    queue.enqueue(
        onHost::FrameSpec{alpaka::Vec{1u}, alpaka::Vec{1u}, exec},
        KernelBundle{IndexTypeKernel<T_Idx>{}, dBuff});
    alpaka::onHost::memcpy(queue, hBuff, dBuff);
    onHost::wait(queue);

    REQUIRE_MESSAGE(hBuff[0] == true, "SharedMdArrayType::index_type failed with index type: " << name);
    REQUIRE_MESSAGE(hBuff[1] == true, "getExtents()::value_type failed with index type: " << name);
}

TEMPLATE_LIST_TEST_CASE("test shared memory index type", "[sharedMem]", TestApis)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);
    alpaka::onHost::Queue queue = device.makeQueue();

    test_index_type<uint32_t>(queue, exec, "uint32_t");
    test_index_type<int64_t>(queue, exec, "int64_t");
    test_index_type<int>(queue, exec, "int");
    test_index_type<size_t>(queue, exec, "size_t");
}
