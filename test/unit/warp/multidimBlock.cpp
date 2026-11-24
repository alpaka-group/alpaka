/* Copyright 2026 Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */

/** @file Tests the warp size and the number of warps in multidimensional blocks against the lanes and warps counted by
 * the threads.
 */

#include "utils.hpp"

#include <alpaka/onAcc/atomic.hpp>
#include <alpaka/onAcc/warp.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <utility>

using namespace alpaka;
using alpaka::test::warp::WarpTestBackends;

struct WarpSizeKernel
{
    ALPAKA_FN_ACC void operator()(
        auto const& acc,
        concepts::IMdSpan<uint32_t> auto reported,
        concepts::IMdSpan<uint32_t> auto counted) const
    {
        // count the lanes of the first warp
        if(onAcc::warp::getWarpIdx(acc) == 0u)
        {
            onAcc::atomicMax(acc, &reported[0u], onAcc::warp::getSize(acc));
            onAcc::atomicAdd(acc, &counted[0u], 1u);
        }
    }
};

struct NumWarpsKernel
{
    ALPAKA_FN_ACC void operator()(
        auto const& acc,
        concepts::IMdSpan<uint32_t> auto reported,
        concepts::IMdSpan<uint32_t> auto counted) const
    {
        // count the warps of the block, each warp has exactly one lane with index zero
        if(onAcc::warp::getLaneIdx(acc) == 0u)
        {
            // the number of warps is one-dimensional on all backends, independent of the dimension of the block
            onAcc::atomicMax(acc, &reported[0u], acc.getExtentsOf(onAcc::origin::block, onAcc::unit::warps).x());
            onAcc::atomicAdd(acc, &counted[0u], 1u);
        }
    }
};

/** Execute a single block
 *
 * @return value reported by alpaka, value counted by the threads
 */
std::pair<uint32_t, uint32_t> executeBlock(auto const& deviceExec, auto const& kernel, auto const& threads)
{
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);
    auto queue = device.makeQueue(queueKind::blocking);

    auto reported = onHost::alloc<uint32_t>(device, 1u);
    auto counted = onHost::alloc<uint32_t>(device, 1u);
    onHost::fill(queue, reported, 0u);
    onHost::fill(queue, counted, 0u);

    // sequential executors support a single thread per block only
    auto const threadsPerBlock = isSeqExecutor(exec) ? threads.fill(1u) : threads;
    queue.enqueue(
        onHost::ThreadSpec{threads.fill(1u), threadsPerBlock, exec},
        KernelBundle{kernel, reported, counted});

    auto reportedHost = onHost::allocHostLike(reported);
    auto countedHost = onHost::allocHostLike(counted);
    onHost::memcpy(queue, reportedHost, reported);
    onHost::memcpy(queue, countedHost, counted);
    onHost::wait(queue);
    return {reportedHost[0u], countedHost[0u]};
}

TEMPLATE_LIST_TEST_CASE("warp size in a multidimensional block", "[warp][multidimBlock]", WarpTestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    // the number of threads is a multiple of the warp size
    std::pair<uint32_t, uint32_t> const result = executeBlock(deviceExec, WarpSizeKernel{}, Vec{4u, 16u});
    uint32_t const warpSize = result.first;
    uint32_t const numLanes = result.second;
    CHECK(warpSize == numLanes);
}

TEMPLATE_LIST_TEST_CASE("number of warps in a multidimensional block", "[warp][multidimBlock]", WarpTestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    for(auto threads : {Vec{4u, 2u}, Vec{7u, 3u}, Vec{4u, 16u}})
    {
        std::pair<uint32_t, uint32_t> const result = executeBlock(deviceExec, NumWarpsKernel{}, threads);
        uint32_t const numWarps = result.first;
        uint32_t const numCountedWarps = result.second;
        INFO("block: " << threads);
        CHECK(numWarps == numCountedWarps);
    }
}
