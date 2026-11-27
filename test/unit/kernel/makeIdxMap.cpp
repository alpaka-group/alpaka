/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace alpaka;

using TestBackends = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

template<onAcc::concepts::IdxTraversing T_Traverse, onAcc::concepts::IdxMapping T_IdxLayout>
struct IotaKernelND
{
    ALPAKA_FN_ACC void operator()(auto const& acc, auto out, auto counter) const
    {
        for(auto i : onAcc::makeIdxMap(
                acc,
                onAcc::worker::threadsInGrid,
                IdxRange{out.getExtents()},
                T_Traverse{},
                T_IdxLayout{}))
        {
            out[i] = i;
            onAcc::atomicAdd(acc, &counter[i], 1u);
        }
    }
};

void testCombination(
    auto queue,
    auto exec,
    concepts::Vector auto extents,
    concepts::Vector auto frameSize,
    auto policy)
{
    auto dBuff = onHost::alloc<ALPAKA_TYPEOF(extents)>(queue.getDevice(), extents);
    auto dBuffCounter = onHost::alloc<uint32_t>(queue.getDevice(), extents);

    auto hBuff = onHost::allocHostLike(dBuff);
    auto hBuffCounter = onHost::allocHostLike(dBuffCounter);
    onHost::wait(queue);
    onHost::memset(queue, dBuff, 0u);
    onHost::memset(queue, dBuffCounter, 0u);

    queue.enqueue(
        onHost::FrameSpec{alpaka::divExZero(extents, frameSize), frameSize, exec},
        KernelBundle{IotaKernelND<ALPAKA_TYPEOF(policy.first), ALPAKA_TYPEOF(policy.second)>{}, dBuff, dBuffCounter});
    onHost::memcpy(queue, hBuff, dBuff);
    onHost::memcpy(queue, hBuffCounter, dBuffCounter);
    onHost::wait(queue);
    // Use REQUIRE instead of CHECK to avoid spamming the output if the results are wrong.
    meta::ndLoopIncIdx(
        extents,
        [&](auto idx)
        {
            REQUIRE(idx == hBuff[idx]);
            // fail if more or less than one invocation on the same data item is detected
            REQUIRE(1u == hBuffCounter[idx]);
        });
}

template<typename T_Queue, typename T_Exec, typename T_Extent, typename T_Policy>
void runTest(T_Queue& queue, T_Exec exec, T_Extent const& extentsAndFrameSize, T_Policy const& policy)
{
    auto bufferExtents = extentsAndFrameSize.first;
    auto frameExtents = extentsAndFrameSize.second;
    DYNAMIC_SECTION(
        "exec=" << onHost::demangledName(exec) << ", extent=" << bufferExtents << ", frameExtents=" << frameExtents
                << ", policy=" << onHost::demangledName<T_Policy>())
    {
        testCombination(queue, exec, bufferExtents, frameExtents, policy);
    }
}

/* The test is using iota to prove that each value in the buffer is set via makeIdxMap.
 * Additionally, we atomically count that each value is only set by a single thread and not twice.
 * We test with different buffer extents, frame extents and policies for makeIdxMap to cover different edge cases of
 * the makeIdxMap implementation.
 */
TEMPLATE_LIST_TEST_CASE("makeIdxMap", "[kernel][makeIdxMap]", TestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);

    onHost::Queue queue = device.makeQueue();

    // list of pairs with (bufferExtents, frameExtents)
    auto extentsAndFrameSize = std::make_tuple(
        // 1D
        std::make_pair(Vec{2u}, Vec{3u}),
        std::make_pair(Vec{7u}, Vec{2u}),
        std::make_pair(Vec{101u}, Vec{101u}),
        std::make_pair(Vec{997u}, Vec{2u}),
        // 2D
        std::make_pair(Vec{2u, 3u}, Vec{5u, 2u}),
        std::make_pair(Vec{7u, 11u}, Vec{2u, 13u}),
        std::make_pair(Vec{17u, 19u}, Vec{23u, 2u}),
        std::make_pair(Vec{101u, 103u}, Vec{2u, 107u}),
        // 3D
        std::make_pair(Vec{2u, 3u, 5u}, Vec{7u, 2u, 3u}),
        std::make_pair(Vec{11u, 13u, 17u}, Vec{2u, 19u, 5u}),
        std::make_pair(Vec{29u, 31u, 37u}, Vec{41u, 2u, 3u}),
        std::make_pair(Vec{43u, 47u, 53u}, Vec{2u, 59u, 5u}),
        // 4D
        std::make_pair(Vec{2u, 3u, 5u, 7u}, Vec{11u, 2u, 3u, 5u}),
        std::make_pair(Vec{11u, 13u, 17u, 19u}, Vec{2u, 23u, 5u, 3u}),
        std::make_pair(Vec{23u, 29u, 31u, 37u}, Vec{41u, 2u, 3u, 5u}),
        std::make_pair(Vec{47u, 43u, 17u, 13u}, Vec{2u, 53u, 19u, 3u}));

    // maybe_unused avoid warnings
    [[maybe_unused]] auto policy = std::make_tuple(
        std::pair<onAcc::traverse::Flat, onAcc::layout::Contiguous>{},
        std::pair<onAcc::traverse::Flat, onAcc::layout::Strided>{},
        std::pair<onAcc::traverse::Tiled, onAcc::layout::Contiguous>{},
        std::pair<onAcc::traverse::Tiled, onAcc::layout::Strided>{});


    auto runExtent = [&](auto const& extent)
    { std::apply([&](auto const&... usedPolicy) { (runTest(queue, exec, extent, usedPolicy), ...); }, policy); };

    std::apply([&](auto const&... extent) { (runExtent(extent), ...); }, extentsAndFrameSize);
}

/** Each worker must get exactly its own index if the worker group and the range describe the same domain. */
struct WorkerIdxKernel
{
    ALPAKA_FN_ACC void operator()(auto const& acc, concepts::IMdSpan<bool> auto success) const
    {
        checkIdx(
            acc,
            success,
            onAcc::worker::blocksInGrid,
            onAcc::range::blocksInGrid,
            acc.getIdxWithin(onAcc::origin::grid, onAcc::unit::blocks));
        checkIdx(
            acc,
            success,
            onAcc::worker::threadsInBlock,
            onAcc::range::threadsInBlock,
            acc.getIdxWithin(onAcc::origin::block, onAcc::unit::threads));
        checkIdx(
            acc,
            success,
            onAcc::worker::linearWarpsInBlock,
            onAcc::range::linearWarpsInBlock,
            Vec{onAcc::warp::getWarpIdx(acc)});
        checkIdx(
            acc,
            success,
            onAcc::worker::linearThreadsInWarp,
            onAcc::range::linearThreadsInWarp,
            Vec{onAcc::warp::getLaneIdx(acc)});
    }

private:
    ALPAKA_FN_ACC void checkIdx(
        auto const& acc,
        concepts::IMdSpan<bool> auto success,
        auto const workGroup,
        auto const range,
        concepts::Vector auto const expectedIdx) const
    {
        uint32_t numIndices = 0u;
        for(auto idx : onAcc::makeIdxMap(acc, workGroup, range))
        {
            if(idx != expectedIdx)
                success[0u] = false;
            ++numIndices;
        }
        // detect missing indices too
        if(numIndices != 1u)
            success[0u] = false;
    }
};

TEMPLATE_LIST_TEST_CASE("makeIdxMap worker index", "[kernel][makeIdxMap]", TestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);

    onHost::Queue queue = device.makeQueue(queueKind::blocking);

    auto successHost = onHost::allocHost<bool>(1u);
    auto successDev = onHost::allocLike(device, successHost);

    auto runFrame = [&](concepts::Vector auto frameExtents)
    {
        onHost::memset(queue, successDev, static_cast<uint8_t>(true));
        queue.enqueue(
            onHost::FrameSpec{frameExtents.fill(3u), frameExtents, exec},
            KernelBundle{WorkerIdxKernel{}, successDev});
        onHost::memcpy(queue, successHost, successDev);
        onHost::wait(queue);
        INFO("frameExtents=" << frameExtents);
        CHECK(successHost[0u]);
    };

    uint32_t const warpSize = device.getDeviceProperties().warpSize;
    // 1D, the last warp is full or contains a single thread
    runFrame(Vec{2u * warpSize});
    runFrame(Vec{2u * warpSize + 1u});
    // MD, warps can be smaller than the warp size
    runFrame(Vec{4u, 2u});
    runFrame(Vec{4u, warpSize + 1u});
    runFrame(Vec{2u, 2u, 8u});
}
