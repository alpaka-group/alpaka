/* Copyright 2026 Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */

/** @file Tests warp::reduce for full and partial warps. */

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
        ALPAKA_FN_ACC void operator()(
            T_Acc const& acc,
            concepts::IMdSpan<bool> auto success,
            uint32_t expectedThreadsPerBlock) const
        {
            uint32_t const warpExtent = onAcc::warp::getSize(acc);
            uint32_t const lane = onAcc::warp::getLaneIdx(acc);
            // warps of a block and blocks are reduced independently
            uint32_t const offset = 100u * onAcc::warp::getWarpIdx(acc) + 1000u * acc[layer::block].idx().x();
            uint32_t const value = lane + 1u + offset;

            auto const numThreads = static_cast<uint32_t>(acc[layer::thread].count().product());
            warpCheck(success, numThreads == expectedThreadsPerBlock);

            // the last warp of a block can be partial, all groups must be complete
            uint32_t const numLanes = onAcc::warp::getNumActiveLanes(acc);
            for(uint32_t width = 1u; width <= warpExtent && numLanes % width == 0u; width *= 2u)
            {
                // result in every lane of the group
                uint32_t const first = lane / width * width;
                warpCheck(
                    success,
                    onAcc::warp::reduce(acc, value, std::plus{}, width) == expectedSum(first, width, offset));
                warpCheck(success, onAcc::warp::reduce(acc, value, MaxOp{}, width) == first + width + offset);

                // floating point: identical result in all lanes of the group
                float const floatSum = onAcc::warp::reduce(acc, 0.1f * static_cast<float>(lane), std::plus{}, width);
                warpCheck(success, onAcc::warp::shfl(acc, floatSum, 0u, width) == floatSum);
            }

            // default width
            if(numLanes == warpExtent)
                warpCheck(success, onAcc::warp::reduce(acc, value, MaxOp{}) == warpExtent + offset);

            // divergent branch: only a part of the warp participates
            uint32_t const half = std::max(1u, warpExtent / 2u);
            if(lane < half && numLanes >= half)
                warpCheck(
                    success,
                    onAcc::warp::reduce(acc, value, std::plus{}, half) == expectedSum(0u, half, offset));
        }
    };

    bool runWarpReduceTest(auto const& deviceExec, alpaka::concepts::Vector auto threads)
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
            KernelBundle{WarpReduceKernel{}, successDev, static_cast<uint32_t>(threadsPerBlock.product())});
        onHost::memcpy(queue, successHost, successDev);
        onHost::wait(queue);
        return successHost[0];
    }
} // namespace

TEMPLATE_LIST_TEST_CASE("warp reduce", "[warp][reduce]", WarpTestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto const warpExtent = test::getDevice(deviceExec).getDeviceProperties().warpSize;

    // full warps, a partial last warp and blocks smaller than a warp
    for(uint32_t const numThreads :
        {4u * warpExtent, 4u * warpExtent - warpExtent / 4u, std::max(1u, warpExtent / 4u)})
    {
        INFO("threads=" << numThreads);
        CHECK(runWarpReduceTest(deviceExec, Vec{numThreads}));
    }

    // multidimensional blocks, a backend can start a new warp with each row of the innermost dimension
    CHECK(runWarpReduceTest(deviceExec, Vec{4u, 15u}));
    CHECK(runWarpReduceTest(deviceExec, Vec{4u, 33u}));
    CHECK(runWarpReduceTest(deviceExec, Vec{4u, warpExtent}));
    CHECK(runWarpReduceTest(deviceExec, Vec{2u, 2u, 8u}));
}
