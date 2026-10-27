/* Copyright 2025 Sergei Bastrakov, Bernhard Manfred Gruber, Jan Stephan, Andrea Bocci, Aurora Perego, Mehmet
 * Yusufoglu, René Widera SPDX-License-Identifier: MPL-2.0
 */

/** @file
 * Tests the warp "getSize" operation which queries the number of threads in a warp.
 * The "getSize" warp operation returns the warp width (e.g., 32 for NVIDIA GPUs, 64 for AMD).
 * It's a query operation that reports the hardware-defined number of threads executing in lockstep.
 */

#include "utils.hpp"

#include <alpaka/onAcc/warp.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>

using namespace alpaka;
using alpaka::test::warp::warpCheck;
using alpaka::test::warp::WarpTestBackends;

namespace
{
    struct GetSizeKernel
    {
        template<typename T_Acc>
        ALPAKA_FN_ACC void operator()(
            T_Acc const& acc,
            concepts::IMdSpan<bool> auto success,
            uint32_t expectedWarpSize,
            uint32_t expectedWarpsPerBlock) const
        {
            // Compare device-reported warp extent against the precomputed traits.
            auto const runtimeSize = onAcc::warp::getSize(acc);
            warpCheck(success, runtimeSize != 0u);
            warpCheck(success, runtimeSize == expectedWarpSize);

            // test if the warp size can be constexpr
            constexpr uint32_t warpExtent = onAcc::warp::getSize<ALPAKA_TYPEOF(acc)>();
            warpCheck(success, warpExtent == expectedWarpSize);
            // laneIdx should be in range [0;warpSize)
            uint32_t const laneIdx = onAcc::warp::getLaneIdx(acc);
            warpCheck(success, laneIdx < warpExtent);

            concepts::Vector auto numTheradsPerWarp = acc.getExtentsOf(onAcc::origin::warp, onAcc::unit::threads);
            warpCheck(success, warpExtent == numTheradsPerWarp.x());

            uint32_t const numWarps = acc[layer::thread].count().product() / warpExtent;
            uint32_t const warpIdx = onAcc::warp::getWarpIdx(acc);
            warpCheck(success, warpIdx < numWarps);

            // Use the linear thread index in the block to calculate the warp index in the block.
            concepts::Vector auto blockThreadCount = acc.getExtentsOf(onAcc::origin::block, onAcc::unit::threads);
            concepts::Vector auto threadIdx = acc.getIdxWithin(onAcc::origin::block, onAcc::unit::threads);
            uint32_t warpIdxByThreadIdx = linearize(blockThreadCount, threadIdx) / warpExtent;
            warpCheck(success, warpIdx == warpIdxByThreadIdx);

            concepts::Vector auto warpIdxInBlock = acc.getIdxWithin(onAcc::origin::block, onAcc::unit::warps);
            warpCheck(success, warpIdx == warpIdxInBlock.x());


            concepts::Vector auto warpIdxInGrid = acc.getIdxWithin(onAcc::origin::grid, onAcc::unit::warps);
            /* Use the block index and the expected number of blocks plus the warp index in the block to get the global
             * warp index */
            warpCheck(
                success,
                acc[layer::block].idx().x() * expectedWarpsPerBlock + warpIdxInBlock.x() == warpIdxInGrid.x());

            // Use the linear global thread index to calculate the warp index.
            concepts::Vector auto threadIdxInGrid = acc.getIdxWithin(onAcc::origin::grid, onAcc::unit::threads);
            concepts::Vector auto threadsInGrid = acc.getExtentsOf(onAcc::origin::grid, onAcc::unit::threads);
            auto linearThreadIdxInGrid = linearize(threadsInGrid, threadIdxInGrid);
            auto linearWarpIdxInGrid = linearThreadIdxInGrid / expectedWarpSize;
            warpCheck(success, linearWarpIdxInGrid == warpIdxInGrid.x());

            concepts::Vector auto numWarpsInBlock = acc.getExtentsOf(onAcc::origin::block, onAcc::unit::warps);
            warpCheck(success, expectedWarpsPerBlock == numWarpsInBlock.x());

            // we started 5 thread blocks each with `expectedWarpsPerBlock` warps
            concepts::Vector auto numWarpsInGrid = acc.getExtentsOf(onAcc::origin::grid, onAcc::unit::warps);
            warpCheck(success, 5u * expectedWarpsPerBlock == numWarpsInGrid.x());
        }
    };
} // namespace

TEMPLATE_LIST_TEST_CASE("warp size trait matches runtime size", "[warp][getSize]", WarpTestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);

    auto deviceProperties = device.getDeviceProperties();
    auto const warpExtent = deviceProperties.warpSize;

    auto queue = device.makeQueue(queueKind::blocking);

    auto successHost = onHost::allocHost<bool>(1u);
    auto successDev = onHost::allocLike(device, successHost);
    auto const blocks = Vec<std::uint32_t, 1u>{5u};

    auto const warpsPerBlock = alpaka::isSeqExecutor(exec) ? 1u : 2u;

    auto const threads = Vec<std::uint32_t, 1u>{warpExtent * warpsPerBlock};

    onHost::memset(queue, successDev, static_cast<std::uint8_t>(true));
    queue.enqueue(
        onHost::ThreadSpec{blocks, threads, exec},
        // Pass the host-side expectation down to the device for verification.
        KernelBundle{GetSizeKernel{}, successDev, static_cast<std::uint32_t>(warpExtent), warpsPerBlock});
    onHost::memcpy(queue, successHost, successDev);
    onHost::wait(queue);
    CHECK(successHost[0]);
}
