/* Copyright 2026 Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */

/** @file Tests warp::inclusiveScan and warp::exclusiveScan for full and partial warps. */

#include "utils.hpp"

#include <alpaka/onAcc/warp.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>

using namespace alpaka;
using alpaka::test::warp::warpCheck;
using alpaka::test::warp::WarpTestBackends;

namespace
{
    // associative but not commutative
    struct TakeFirst
    {
        constexpr std::uint32_t operator()(std::uint32_t first, std::uint32_t) const
        {
            return first;
        }
    };

    // sum of v(lane) = lane + 1 + offset over [first, first + count)
    constexpr std::uint32_t expectedSum(std::uint32_t first, std::uint32_t count, std::uint32_t offset)
    {
        return count * (first + 1u + offset) + count * (count - 1u) / 2u;
    }

    struct WarpScanKernel
    {
        template<typename T_Acc>
        ALPAKA_FN_ACC void operator()(
            T_Acc const& acc,
            concepts::IMdSpan<bool> auto success,
            uint32_t expectedThreadsPerBlock) const
        {
            // the requested block size must be used, else the full and partial warp cases are not tested
            warpCheck(success, static_cast<uint32_t>(acc[layer::thread].count().product()) == expectedThreadsPerBlock);

            uint32_t const warpExtent = onAcc::warp::getSize(acc);
            uint32_t const lane = onAcc::warp::getLaneIdx(acc);
            // warps of a block and blocks are scanned independently
            uint32_t const offset = 100u * onAcc::warp::getWarpIdx(acc) + 1000u * acc[layer::block].idx().x();
            uint32_t const value = lane + 1u + offset;

            // every lane gets the scan over the lanes up to it
            warpCheck(
                success,
                onAcc::warp::inclusiveScan(acc, value, std::plus{}) == expectedSum(0u, lane + 1u, offset));
            warpCheck(
                success,
                onAcc::warp::exclusiveScan(acc, value, std::plus{}, 7u)
                    == (lane == 0u ? 7u : expectedSum(0u, lane, offset)));

            for(uint32_t width = 1u; width <= warpExtent; width *= 2u)
            {
                uint32_t const first = lane / width * width;
                warpCheck(
                    success,
                    onAcc::warp::inclusiveScan(acc, value, std::plus{}, width)
                        == expectedSum(first, lane - first + 1u, offset));
                warpCheck(
                    success,
                    onAcc::warp::exclusiveScan(acc, value, std::plus{}, 7u, width)
                        == (lane == first ? 7u : expectedSum(first, lane - first, offset)));
                // order of operands: non commutative op, every lane gets the value of the first lane of its group
                warpCheck(success, onAcc::warp::inclusiveScan(acc, value, TakeFirst{}, width) == offset + first + 1u);
            }

            // divergent branch: only a part of the warp participates
            uint32_t const half = std::max(1u, warpExtent / 2u);
            if(lane < half)
                warpCheck(
                    success,
                    onAcc::warp::inclusiveScan(acc, value, std::plus{}, half) == expectedSum(0u, lane + 1u, offset));
        }
    };

    bool runWarpScanTest(auto const& deviceExec, alpaka::concepts::Vector auto threads)
    {
        onHost::Device device = test::getDevice(deviceExec);
        concepts::Executor auto exec = test::getExecutor(deviceExec);
        auto queue = device.makeQueue(queueKind::blocking);

        auto successHost = onHost::allocHost<bool>(1u);
        auto successDev = onHost::allocLike(device, successHost);
        onHost::memset(queue, successDev, static_cast<std::uint8_t>(true));

        // sequential executors support a single thread per block only
        auto const threadsPerBlock = alpaka::isSeqExecutor(exec) ? threads.fill(1u) : threads;
        queue.enqueue(
            onHost::ThreadSpec{threadsPerBlock.fill(3u), threadsPerBlock, exec},
            KernelBundle{WarpScanKernel{}, successDev, static_cast<uint32_t>(threadsPerBlock.product())});
        onHost::memcpy(queue, successHost, successDev);
        onHost::wait(queue);
        return successHost[0];
    }
} // namespace

TEMPLATE_LIST_TEST_CASE("warp scan", "[warp][scan]", WarpTestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto const warpExtent = test::getDevice(deviceExec).getDeviceProperties().warpSize;

    // full warps, a partial last warp and blocks smaller than a warp
    for(uint32_t const numThreads :
        {4u * warpExtent, 4u * warpExtent - warpExtent / 4u, std::max(1u, warpExtent / 4u)})
    {
        INFO("threads=" << numThreads);
        CHECK(runWarpScanTest(deviceExec, Vec{numThreads}));
    }

    // multidimensional blocks, a backend can start a new warp with each row of the innermost dimension
    CHECK(runWarpScanTest(deviceExec, Vec{4u, 15u}));
    CHECK(runWarpScanTest(deviceExec, Vec{4u, 33u}));
    CHECK(runWarpScanTest(deviceExec, Vec{4u, warpExtent}));
    CHECK(runWarpScanTest(deviceExec, Vec{2u, 2u, 8u}));
}
