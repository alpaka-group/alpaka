/* Copyright 2026 Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */

/** @file Tests warp::inclusiveScan and warp::exclusiveScan for full warps and blocks smaller than a warp. */

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
        ALPAKA_FN_ACC void operator()(T_Acc const& acc, concepts::IMdSpan<bool> auto success) const
        {
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

            // order of operands: non commutative op, every lane gets the value of lane 0
            warpCheck(success, onAcc::warp::inclusiveScan(acc, value, TakeFirst{}) == offset + 1u);

            // branch splitting the warp: select the value in the branch, scan with all lanes
            uint32_t const split = warpExtent < 4u ? warpExtent : 4u;
            uint32_t const scanned = onAcc::warp::inclusiveScan(acc, lane < split ? 0u : value, std::plus{});
            warpCheck(success, scanned == (lane < split ? 0u : expectedSum(split, lane - split + 1u, offset)));
        }
    };
} // namespace

TEMPLATE_LIST_TEST_CASE("warp scan", "[warp][scan]", WarpTestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);
    auto const warpExtent = device.getDeviceProperties().warpSize;
    auto queue = device.makeQueue(queueKind::blocking);

    auto successHost = onHost::allocHost<bool>(1u);
    auto successDev = onHost::allocLike(device, successHost);
    // blocks with several warps and blocks smaller than a warp
    for(uint32_t const numThreads : {4u * warpExtent, std::max(1u, warpExtent / 4u)})
    {
        onHost::memset(queue, successDev, static_cast<std::uint8_t>(true));
        queue.enqueue(onHost::FrameSpec{Vec{3u}, Vec{numThreads}, exec}, KernelBundle{WarpScanKernel{}, successDev});
        onHost::memcpy(queue, successHost, successDev);
        onHost::wait(queue);
        CHECK(successHost[0]);
    }
}
