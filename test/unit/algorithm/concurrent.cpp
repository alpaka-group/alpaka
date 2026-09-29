/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/meta/CartesianProduct.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <functional>
#include <type_traits>

using namespace alpaka;

using TestBackends = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

/** Stencil SIMD functor
 *
 * The stencil functor is getting a SimdPtr as input which allows to call the operator[] to change the index.
 * In this test we do not shift the memory to avoid out of memory access.
 * We cannot use generic lambdas with CUDA therefore we need to write a functor.
 */
struct StencilAddWithAcc
{
    constexpr void operator()(
        onAcc::concepts::Acc auto const&,
        concepts::SimdPtr auto a,
        concepts::SimdPtr auto const& b) const
    {
        a = a.load() + b.load();
    }
};

struct StencilAddMin
{
    constexpr void operator()(concepts::SimdPtr auto a, concepts::SimdPtr auto const& b) const
    {
        constexpr uint32_t simdWidth = a.width();
        alpaka::concepts::Simd auto aSimd = a.load();
        alpaka::concepts::Simd auto bSimd = b.load();
        for(uint32_t idx = 0u; idx < simdWidth; ++idx)
        {
            // SIMD comparisons results in a SimdMask, accessing single lanes of a mask are allowed to return a value
            // wrapper, there we need to use the unary + operator to cast the wrapper to a value
            aSimd[idx] += math::min(+aSimd[idx], +bSimd[idx]);
        }
        // write result back to a
        a = aSimd;
    }
};

struct TestWithMdSpan
{
    /**
     * @param setup all functors should write the results to the first argument
     */
    template<typename T_DataType>
    static void executeTest(
        concepts::Executor auto exec,
        auto const& computeQueue,
        auto const setup,
        concepts::Vector auto extentMd)
    {
        INFO("run func : " << onHost::demangledName(std::get<1>(setup)));

        auto computeDev = computeQueue.getDevice();
        using DataType = T_DataType;
        onHost::SharedBuffer computeBufferIn0 = onHost::alloc<DataType>(computeDev, extentMd);
        onHost::SharedBuffer computeBufferIn1 = onHost::allocLike(computeDev, computeBufferIn0);
        onHost::SharedBuffer hostBufferIota = onHost::allocLike(onHost::makeHostDevice(), computeBufferIn0);
        onHost::SharedBuffer hostBufferOut = onHost::allocLike(onHost::makeHostDevice(), computeBufferIn0);

        // initialize with the linearized index
        DataType iotaCounter = 0;
        for(auto& value : hostBufferIota)
        {
            value = iotaCounter;
            ++iotaCounter;
        }

        onHost::memcpy(computeQueue, computeBufferIn0, hostBufferIota);
        onHost::memcpy(computeQueue, computeBufferIn1, hostBufferIota);

        onHost::wait(computeQueue);
        auto const beginT = std::chrono::high_resolution_clock::now();

        onHost::concurrent<DataType>(
            computeQueue,
            exec,
            extentMd,
            std::get<0>(setup),
            computeBufferIn0,
            computeBufferIn1);

        onHost::wait(computeQueue);
        auto const endT = std::chrono::high_resolution_clock::now();
        UNSCOPED_INFO(
            "Time for concurrent: " << std::chrono::duration<double>(endT - beginT).count()
                                    << "s data size: " << extentMd);

        onHost::memcpy(computeQueue, hostBufferOut, computeBufferIn0);
        onHost::wait(computeQueue);

        // validate without using the forward iterator
        DataType refIotaCounter = 0;
        meta::ndLoopIncIdx(
            extentMd,
            [&](auto idx)
            {
                CHECK(hostBufferOut[idx] == std::get<1>(setup)(refIotaCounter, refIotaCounter));
                ++refIotaCounter;
            });
    }
};

template<typename T_DataType>
void prepareTest(auto cfg, concepts::Vector auto extentMd, auto const& setupTuple)
{
    using DataType = T_DataType;
    auto deviceExec = test::getDeviceExecutorOrSkipTest(cfg);
    onHost::Device computeDev = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);

    onHost::Queue computeQueue = computeDev.makeQueue();

    // execute for each functor
    std::apply(
        [&](auto const&... setup)
        { (std::get<2>(setup).template executeTest<DataType>(exec, computeQueue, setup, extentMd), ...); },
        setupTuple);
}

TEMPLATE_LIST_TEST_CASE("concurrent", "", TestBackends)
{
    auto cfg = TestType::makeDict();

    using DataType = int;

    // This list is not directly defined within the function `prepareTest()` due to nvcc compile issues.
    auto setups = std::make_tuple(
        std::make_tuple(StencilAddWithAcc{}, std::plus{}, TestWithMdSpan{}),
        std::make_tuple(
            StencilAddMin{},
            [](DataType const& a, DataType const& b) { return a + math::min(a, b); },
            TestWithMdSpan{}));

    // different extents for testing
    auto extentMdList
        = std::make_tuple(Vec{5, 7, 3, 11}, Vec{93, 7, 123}, Vec{5, 7, 4111}, Vec{5, 7, 3}, Vec{7, 3}, Vec{3});

    std::apply([&](auto... extents) { (prepareTest<DataType>(cfg, extents, setups), ...); }, extentMdList);
}

struct TestWithGenerator
{
    /**
     * @param setup all functors should write the results to the first argument
     */
    template<typename T_DataType>
    static void executeTest(
        concepts::Executor auto exec,
        auto const& computeQueue,
        auto const setup,
        concepts::Vector auto extentMd)
    {
        INFO("run func : " << onHost::demangledName(std::get<1>(setup)));

        auto computeDev = computeQueue.getDevice();
        using DataType = T_DataType;
        onHost::SharedBuffer computeBufferIn0 = onHost::alloc<DataType>(computeDev, extentMd);
        onHost::SharedBuffer hostBufferIota = onHost::allocLike(onHost::makeHostDevice(), computeBufferIn0);
        onHost::SharedBuffer hostBufferOut = onHost::allocLike(onHost::makeHostDevice(), computeBufferIn0);

        auto generator = LinearizedIdxGenerator{extentMd};

        // initialize with the linearized index
        DataType iotaCounter = 0;
        for(auto& value : hostBufferIota)
        {
            value = iotaCounter;
            ++iotaCounter;
        }

        onHost::memcpy(computeQueue, computeBufferIn0, hostBufferIota);

        onHost::wait(computeQueue);
        auto const beginT = std::chrono::high_resolution_clock::now();

        onHost::concurrent<DataType>(computeQueue, exec, extentMd, std::get<0>(setup), computeBufferIn0, generator);

        onHost::wait(computeQueue);
        auto const endT = std::chrono::high_resolution_clock::now();
        UNSCOPED_INFO(
            "Time for concurrent: " << std::chrono::duration<double>(endT - beginT).count()
                                    << "s data size: " << extentMd);

        onHost::memcpy(computeQueue, hostBufferOut, computeBufferIn0);
        onHost::wait(computeQueue);

        // validate without using the forward iterator
        DataType refIotaCounter = 0;
        meta::ndLoopIncIdx(
            extentMd,
            [&](auto idx)
            {
                CHECK(hostBufferOut[idx] == std::get<1>(setup)(refIotaCounter, generator[idx]));
                ++refIotaCounter;
            });
    }
};

TEMPLATE_LIST_TEST_CASE("concurrent generator", "", TestBackends)
{
    auto cfg = TestType::makeDict();

    using DataType = int;

    // This list is not directly defined within the function `prepareTest()` due to nvcc compile issues.
    auto setups = std::make_tuple(
        std::make_tuple(StencilAddWithAcc{}, std::plus{}, TestWithGenerator{}),
        std::make_tuple(
            StencilAddMin{},
            [](DataType const& a, DataType const& b) { return a + math::min(a, b); },
            TestWithGenerator{}));

    // different extents for testing
    auto extentMdList
        = std::make_tuple(Vec{5, 7, 3, 11}, Vec{93, 7, 123}, Vec{5, 7, 4111}, Vec{5, 7, 3}, Vec{7, 3}, Vec{3});

    std::apply([&](auto... extents) { (prepareTest<DataType>(cfg, extents, setups), ...); }, extentMdList);
}
