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

namespace myTest
{
    // This functor does not support Simd and must be wrapped by ScalarFunc
    struct MinValue
    {
        constexpr auto operator()(auto const& a, auto const& b) const
        {
            return math::min(a, b);
        }
    };

    ALPAKA_FN_ACC void alpakaAtomicInvoke(
        MinValue const&,
        alpaka::onAcc::concepts::Acc auto const& acc,
        auto* dest,
        auto const src)
    {
        onAcc::atomicMin(acc, dest, src);
    }

    // This functor does not support Simd and must be wrapped by ScalarFunc
    struct MaxValue
    {
        constexpr auto operator()(auto const& a, auto const& b) const
        {
            return math::max(a, b);
        }
    };

    ALPAKA_FN_ACC void alpakaAtomicInvoke(
        MaxValue const&,
        alpaka::onAcc::concepts::Acc auto const& acc,
        auto* dest,
        auto const src)
    {
        onAcc::atomicMax(acc, dest, src);
    }
} // namespace myTest

struct TestWithMdSpan
{
    template<typename T_DataType>
    static void executeTest(
        concepts::Executor auto exec,
        auto const& computeQueue,
        auto const setup,
        concepts::Vector auto extentMd)
    {
        INFO("run func : " << onHost::demangledName(std::get<2>(setup)));

        auto computeDev = computeQueue.getDevice();
        using DataType = T_DataType;
        using OutDataType = ALPAKA_TYPEOF(std::get<0>(setup));
        onHost::SharedBuffer computeBufferOut = onHost::allocDeferred<OutDataType>(computeQueue, extentMd.fill(1));
        onHost::SharedBuffer computeBufferIn = onHost::allocDeferred<DataType>(computeQueue, extentMd);
        onHost::SharedBuffer hostBufferIota = onHost::allocHostLike(computeBufferIn);
        onHost::SharedBuffer hostBufferOut = onHost::allocHostLike(computeBufferOut);

        // initialize with the linearized index
        DataType iotaCounter = 0;
        for(auto& value : hostBufferIota)
        {
            value = iotaCounter;
            ++iotaCounter;
        }

        onHost::memcpy(computeQueue, computeBufferIn, hostBufferIota);

        onHost::wait(computeQueue);
        auto const beginT = std::chrono::high_resolution_clock::now();

        onHost::reduce(computeQueue, exec, std::get<0>(setup), computeBufferOut, std::get<1>(setup), computeBufferIn);

        onHost::wait(computeQueue);
        auto const endT = std::chrono::high_resolution_clock::now();
        UNSCOPED_INFO(
            "Time for reduction: " << std::chrono::duration<double>(endT - beginT).count()
                                   << "s data size: " << hostBufferIota.getExtents());

        onHost::memcpy(computeQueue, hostBufferOut, computeBufferOut);
        onHost::wait(computeQueue);

        // validate without using the forward iterator
        DataType refIotaCounter = 0;
        auto result = std::get<0>(setup);
        meta::ndLoopIncIdx(
            extentMd,
            [&](auto idx)
            {
                alpaka::unused(idx);
                result = std::get<2>(setup)(result, refIotaCounter);
                ++refIotaCounter;
            });

        CHECK(*hostBufferOut.data() == result);
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
        { (std::get<3>(setup).template executeTest<DataType>(exec, computeQueue, setup, extentMd), ...); },
        setupTuple);
}

TEMPLATE_LIST_TEST_CASE("reduce", "", TestBackends)
{
    auto cfg = TestType::makeDict();

    using DataType = int;

    // This list is not directly defined within the function TestWithMdSpan due to nvcc compile issues.
    auto setups = std::make_tuple(
        std::make_tuple(DataType{0}, std::plus{}, std::plus{}, TestWithMdSpan{}),
        std::make_tuple(
            std::numeric_limits<DataType>::max(),
            ScalarFunc{myTest::MinValue{}},
            myTest::MinValue{},
            TestWithMdSpan{}),
        std::make_tuple(
            std::numeric_limits<DataType>::min(),
            ScalarFunc{myTest::MaxValue{}},
            myTest::MaxValue{},
            TestWithMdSpan{}));

    // different extents for testing
    auto extentMdList
        = std::make_tuple(Vec{5, 7, 3, 11}, Vec{93, 7, 123}, Vec{5, 7, 4111}, Vec{5, 7, 3}, Vec{7, 3}, Vec{3});

    std::apply([&](auto... extents) { (prepareTest<DataType>(cfg, extents, setups), ...); }, extentMdList);
}

struct TestWithGenerator
{
    template<typename T_DataType>
    static void executeTest(
        concepts::Executor auto exec,
        auto const& computeQueue,
        auto const functorPair,
        concepts::Vector auto extentMd)
    {
        INFO("run func : " << onHost::demangledName(std::get<2>(functorPair)));

        auto computeDev = computeQueue.getDevice();
        using OutDataType = ALPAKA_TYPEOF(std::get<0>(functorPair));
        onHost::SharedBuffer computeBufferOut = onHost::allocDeferred<OutDataType>(computeQueue, extentMd.fill(1));
        auto generator = LinearizedIdxGenerator{extentMd};

        onHost::SharedBuffer hostBufferOut = onHost::allocHostLike(computeBufferOut);

        onHost::wait(computeQueue);
        auto const beginT = std::chrono::high_resolution_clock::now();

        onHost::reduce(
            computeQueue,
            exec,
            std::get<0>(functorPair),
            computeBufferOut,
            std::get<1>(functorPair),
            generator);

        onHost::wait(computeQueue);
        auto const endT = std::chrono::high_resolution_clock::now();
        UNSCOPED_INFO(
            "Time for reduction: " << std::chrono::duration<double>(endT - beginT).count()
                                   << "s data size: " << generator.getExtents());

        onHost::memcpy(computeQueue, hostBufferOut, computeBufferOut);
        onHost::wait(computeQueue);

        auto result = std::get<0>(functorPair);
        meta::ndLoopIncIdx(extentMd, [&](auto idx) { result = std::get<2>(functorPair)(result, generator[idx]); });

        CHECK(*hostBufferOut.data() == result);
    }
};

TEMPLATE_LIST_TEST_CASE("reduce generator", "", TestBackends)
{
    auto cfg = TestType::makeDict();

    using DataType = int;

    // This list is not directly defined within the function TestWithGenerator due to nvcc compile issues.
    auto setup = std::make_tuple(
        std::make_tuple(DataType{0}, std::plus{}, std::plus{}, TestWithGenerator{}),
        std::make_tuple(
            std::numeric_limits<DataType>::max(),
            ScalarFunc{myTest::MinValue{}},
            myTest::MinValue{},
            TestWithGenerator{}),
        std::make_tuple(
            std::numeric_limits<DataType>::min(),
            ScalarFunc{myTest::MaxValue{}},
            myTest::MaxValue{},
            TestWithGenerator{}));

    // different extents for testing
    auto extentMdList
        = std::make_tuple(Vec{5, 7, 3, 11}, Vec{93, 7, 123}, Vec{5, 7, 4111}, Vec{5, 7, 3}, Vec{7, 3}, Vec{3});

    std::apply([&](auto... extents) { (prepareTest<DataType>(cfg, extents, setup), ...); }, extentMdList);
}
