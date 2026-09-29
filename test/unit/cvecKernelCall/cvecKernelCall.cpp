/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <functional>
#include <thread>

/** @file In this file we test if the a compile time thread extent for number of threads within a block is forwarded
 * correctly into the kernel without silent conversions into a runtime value.
 */

using namespace alpaka;
using namespace alpaka::onHost;

using TestApis = std::decay_t<decltype(allBackends(enabledApis, exec::enabledExecutors))>;

struct KernelCVecFrameExtents
{
    template<onAcc::concepts::Acc T_Acc>
    ALPAKA_FN_ACC void operator()(T_Acc const& acc, auto result) const
    {
        static_assert(T_Acc::launchedWithFrameSpec());
        // compile will fail if the type is silently cast to non CVec type
        [[maybe_unused]] alpaka::concepts::CVector auto blockThreadCount
            = acc.getExtentsOf(onAcc::origin::block, onAcc::unit::threads);
        [[maybe_unused]] alpaka::concepts::CVector auto blockThreadCount2 = acc[alpaka::layer::thread].count();
        static_assert(blockThreadCount2 == blockThreadCount);

        // warp size must be accessible at compile time, this is only possible if we use the accelerator type
        constexpr uint32_t warpSize = ALPAKA_TYPEOF(acc)::getWarpSize();
        constexpr uint32_t warpSizeFn = onAcc::warp::getSize<T_Acc>();

        static_assert(warpSize == warpSizeFn);
        result[0] = true;
    }
};

TEMPLATE_LIST_TEST_CASE("CVec frame extent kernel call", "", TestApis)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    alpaka::concepts::Executor auto exec = test::getExecutor(deviceExec);
    Queue queue = device.makeQueue();

    auto dBuff = onHost::alloc<bool>(device, Vec{1u});

    auto hBuff = onHost::allocHostLike(dBuff);
    onHost::wait(queue);
    {
        queue.enqueue(
            FrameSpec{Vec{1u}, CVec<uint32_t, 43u>{}, exec},
            KernelBundle{KernelCVecFrameExtents{}, dBuff.getView()});
        onHost::memcpy(queue, hBuff, dBuff);
        onHost::wait(queue);
        CHECK(hBuff[0] == true);
    }
}

struct KernelCVecThreadExtents
{
    template<onAcc::concepts::Acc T_Acc>
    ALPAKA_FN_ACC void operator()(T_Acc const& acc, auto result) const
    {
        static_assert(!T_Acc::launchedWithFrameSpec());
        // compile will fail if the type is silently cast to non CVec type
        alpaka::concepts::CVector auto blockThreadCount = acc[alpaka::layer::thread].count();
        static_assert(blockThreadCount.x() == 1u);
        result[0] = blockThreadCount.x() == 1u;

        // warp size must be accessible at compile time, this is only possible if we use the accelerator type
        constexpr uint32_t warpSize = ALPAKA_TYPEOF(acc)::getWarpSize();
        constexpr uint32_t warpSizeFn = onAcc::warp::getSize<T_Acc>();

        static_assert(warpSize == warpSizeFn);
    }
};

TEMPLATE_LIST_TEST_CASE("CVec thread extent kernel call", "", TestApis)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    alpaka::concepts::Executor auto exec = test::getExecutor(deviceExec);
    Queue queue = device.makeQueue();

    auto dBuff = onHost::alloc<bool>(device, Vec{1u});

    auto hBuff = onHost::allocHostLike(dBuff);
    onHost::wait(queue);
    {
        queue.enqueue(
            // we need to use one thread per thread block because serial executors will reduce the number of threads to
            // a single thread
            ThreadSpec{Vec{1u}, CVec<uint32_t, 1u>{}, exec},
            KernelBundle{KernelCVecThreadExtents{}, dBuff});
        onHost::memcpy(queue, hBuff, dBuff);
        onHost::wait(queue);
        CHECK(hBuff[0] == true);
    }
}
