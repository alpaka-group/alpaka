/* Copyright 2026 Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */

/** @file Tests warp::reduce for full warps and blocks smaller than a warp. */

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
    struct MaxOp
    {
        constexpr std::uint32_t operator()(std::uint32_t a, std::uint32_t b) const
        {
            return a > b ? a : b;
        }
    };

    // sum of v(lane) = lane + 1 + offset over [first, first + count)
    constexpr std::uint32_t expectedSum(std::uint32_t first, std::uint32_t count, std::uint32_t offset)
    {
        return count * (first + 1u + offset) + count * (count - 1u) / 2u;
    }

    struct WarpReduceKernel
    {
        template<typename T_Acc>
        ALPAKA_FN_ACC void operator()(T_Acc const& acc, concepts::IMdSpan<bool> auto success) const
        {
            uint32_t const warpExtent = onAcc::warp::getSize(acc);
            uint32_t const lane = onAcc::warp::getLaneIdx(acc);
            // warps of a block and blocks are reduced independently
            uint32_t const offset = 100u * onAcc::warp::getWarpIdx(acc) + 1000u * acc[layer::block].idx().x();
            uint32_t const value = lane + 1u + offset;

            // a block can be smaller than a warp
            uint32_t const numLanes
                = std::min(warpExtent, static_cast<uint32_t>(acc[layer::thread].count().product()));

            // result in every lane
            warpCheck(success, onAcc::warp::reduce(acc, value, std::plus{}) == expectedSum(0u, numLanes, offset));
            warpCheck(success, onAcc::warp::reduce(acc, value, MaxOp{}) == numLanes + offset);

            // floating point: identical result in all lanes
            float const floatSum = onAcc::warp::reduce(acc, 0.1f * static_cast<float>(lane), std::plus{});
            warpCheck(success, onAcc::warp::shfl(acc, floatSum, 0u) == floatSum);

            // branch splitting the warp: select the value in the branch, reduce with all lanes
            uint32_t const split = numLanes < 4u ? numLanes : 4u;
            uint32_t const sumLow = onAcc::warp::reduce(acc, lane < split ? value : 0u, std::plus{});
            uint32_t const sumHigh = onAcc::warp::reduce(acc, lane < split ? 0u : value, std::plus{});
            warpCheck(success, sumLow == expectedSum(0u, split, offset));
            warpCheck(success, sumHigh == expectedSum(split, numLanes - split, offset));
        }
    };
} // namespace

TEMPLATE_LIST_TEST_CASE("warp reduce", "[warp][reduce]", WarpTestBackends)
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
        queue.enqueue(onHost::FrameSpec{Vec{3u}, Vec{numThreads}, exec}, KernelBundle{WarpReduceKernel{}, successDev});
        onHost::memcpy(queue, successHost, successDev);
        onHost::wait(queue);
        CHECK(successHost[0]);
    }
}
