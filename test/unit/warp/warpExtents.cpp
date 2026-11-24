/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

/** @file Tests the number of active lanes per warp and the number of warps per block and grid.
 *
 * Warps are 1D groups of threads. The number of threads inside a block is not required to be a multiple of the warp
 * size, therefore the last warp can have fewer lanes. On SYCL CPU sub-groups can also be smaller than the requested
 * warp size and a multidimensional block can contain more sub-groups than a linear chunking would produce.
 */

#include "utils.hpp"

#include <alpaka/onAcc/atomic.hpp>
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
    /** Verify all warp extents for a 1D block, where the whole block is a linear sequence of warps. */
    struct LinearWarpExtentsKernel
    {
        ALPAKA_FN_ACC void operator()(auto const& acc, concepts::IMdSpan<bool> auto success, uint32_t blocksInGrid)
            const
        {
            constexpr uint32_t warpExtent = onAcc::warp::getSize<ALPAKA_TYPEOF(acc)>();
            auto const blockExtents = acc.getExtentsOf(onAcc::origin::block, onAcc::unit::threads);
            auto const blockThreads = static_cast<uint32_t>(blockExtents.product());
            auto const linearThreadIdx = static_cast<uint32_t>(
                linearize(blockExtents, acc.getIdxWithin(onAcc::origin::block, onAcc::unit::threads)));

            uint32_t const expectedWarpIdx = linearThreadIdx / warpExtent;
            uint32_t const remainingThreads = blockThreads - expectedWarpIdx * warpExtent;
            uint32_t const expectedLanes = remainingThreads < warpExtent ? remainingThreads : warpExtent;
            uint32_t const expectedNumWarps = static_cast<uint32_t>(alpaka::divCeil(blockThreads, warpExtent));
            uint32_t const blockIdx = static_cast<uint32_t>(acc[layer::block].idx().x());
            uint32_t const warpIdx = onAcc::warp::getWarpIdx(acc);
            uint32_t const laneIdx = onAcc::warp::getLaneIdx(acc);
            uint32_t const numActiveLanes = onAcc::warp::getNumActiveLanes(acc);

            warpCheck(success, laneIdx == linearThreadIdx % warpExtent);
            warpCheck(success, warpIdx == expectedWarpIdx);
            warpCheck(success, numActiveLanes == expectedLanes);
            warpCheck(success, laneIdx < numActiveLanes);
            warpCheck(success, acc.getExtentsOf(onAcc::origin::warp, onAcc::unit::threads).x() == expectedLanes);
            warpCheck(success, onAcc::warp::getNumWarps(acc) == expectedNumWarps);
            warpCheck(success, acc.getExtentsOf(onAcc::origin::block, onAcc::unit::warps).x() == expectedNumWarps);
            warpCheck(
                success,
                acc.getExtentsOf(onAcc::origin::grid, onAcc::unit::warps).x() == blocksInGrid * expectedNumWarps);
            warpCheck(
                success,
                acc.getIdxWithin(onAcc::origin::grid, onAcc::unit::warps).x()
                    == blockIdx * expectedNumWarps + warpIdx);
        }
    };

    /** Count the lanes and warps of multidimensional blocks and verify the reported extents against the counts.
     *
     * The number of sub-groups for a multidimensional block is backend specific, e.g. SYCL CPU forms one sub-group per
     * innermost row. Therefore the values are not hard-coded but compared against the lanes and warps counted by the
     * threads.
     */
    struct MdWarpExtentsKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            concepts::IMdSpan<bool> auto success,
            concepts::IMdSpan<uint32_t> auto reportedLanes,
            concepts::IMdSpan<uint32_t> auto countedLanes,
            uint32_t stride,
            uint32_t blocksInGrid) const
        {
            uint32_t const blockIdx = static_cast<uint32_t>(linearize(
                acc.getExtentsOf(onAcc::origin::grid, onAcc::unit::blocks),
                acc.getIdxWithin(onAcc::origin::grid, onAcc::unit::blocks)));
            uint32_t const warpIdx = onAcc::warp::getWarpIdx(acc);
            uint32_t const laneIdx = onAcc::warp::getLaneIdx(acc);
            uint32_t const numActiveLanes = onAcc::warp::getNumActiveLanes(acc);
            uint32_t const numWarps = onAcc::warp::getNumWarps(acc);
            uint32_t const slot = blockIdx * stride + warpIdx;

            onAcc::atomicMax(acc, &reportedLanes[slot], numActiveLanes);
            onAcc::atomicAdd(acc, &countedLanes[slot], 1u);

            warpCheck(success, warpIdx < numWarps);
            warpCheck(success, laneIdx < numActiveLanes);
            warpCheck(success, numWarps == acc.getExtentsOf(onAcc::origin::block, onAcc::unit::warps).x());
            warpCheck(success, acc.getExtentsOf(onAcc::origin::warp, onAcc::unit::threads).x() == numActiveLanes);
            warpCheck(
                success,
                acc.getExtentsOf(onAcc::origin::grid, onAcc::unit::warps).x() == blocksInGrid * numWarps);
            warpCheck(
                success,
                acc.getIdxWithin(onAcc::origin::grid, onAcc::unit::warps).x() == blockIdx * numWarps + warpIdx);
        }
    };

    template<typename TKernel>
    bool runLinearExtentsTest(auto const& deviceExec, Vec<uint32_t, 1u> blocks, Vec<uint32_t, 1u> threads)
    {
        onHost::Device device = test::getDevice(deviceExec);
        concepts::Executor auto exec = test::getExecutor(deviceExec);
        auto queue = device.makeQueue(queueKind::blocking);

        auto successHost = onHost::allocHost<bool>(1u);
        auto successDev = onHost::allocLike(device, successHost);
        onHost::memset(queue, successDev, static_cast<uint8_t>(true));

        // sequential executors support a single thread per block only
        auto const threadsPerBlock = isSeqExecutor(exec) ? threads.fill(1u) : threads;
        queue.enqueue(
            onHost::ThreadSpec{blocks, threadsPerBlock, exec},
            KernelBundle{TKernel{}, successDev, blocks[0u]});

        onHost::memcpy(queue, successHost, successDev);
        onHost::wait(queue);
        return successHost[0u];
    }

    bool runMdExtentsTest(auto const& deviceExec, auto threads)
    {
        onHost::Device device = test::getDevice(deviceExec);
        concepts::Executor auto exec = test::getExecutor(deviceExec);
        auto queue = device.makeQueue(queueKind::blocking);

        auto const threadsPerBlock = isSeqExecutor(exec) ? threads.fill(1u) : threads;
        auto const stride = static_cast<uint32_t>(threadsPerBlock.product());
        auto const blockExtents = threadsPerBlock.fill(3u);
        auto const numBlocks = static_cast<uint32_t>(blockExtents.product());

        auto successHost = onHost::allocHost<bool>(1u);
        auto successDev = onHost::allocLike(device, successHost);
        auto reportedDev = onHost::alloc<uint32_t>(device, numBlocks * stride);
        auto countedDev = onHost::alloc<uint32_t>(device, numBlocks * stride);
        onHost::memset(queue, successDev, static_cast<uint8_t>(true));
        onHost::fill(queue, reportedDev, 0u);
        onHost::fill(queue, countedDev, 0u);

        queue.enqueue(
            onHost::ThreadSpec{blockExtents, threadsPerBlock, exec},
            KernelBundle{MdWarpExtentsKernel{}, successDev, reportedDev, countedDev, stride, numBlocks});

        auto reportedHost = onHost::allocHostLike(reportedDev);
        auto countedHost = onHost::allocHostLike(countedDev);
        onHost::memcpy(queue, reportedHost, reportedDev);
        onHost::memcpy(queue, countedHost, countedDev);
        onHost::memcpy(queue, successHost, successDev);
        onHost::wait(queue);

        bool result = successHost[0u];
        for(uint32_t block = 0u; block < numBlocks; ++block)
        {
            uint32_t threadsInBlock = 0u;
            for(uint32_t warp = 0u; warp < stride; ++warp)
            {
                auto const idx = block * stride + warp;
                uint32_t const counted = countedHost[idx];
                if(counted == 0u)
                    continue;
                threadsInBlock += counted;
                // all threads of a warp see the same number of lanes, which must match the counted lanes
                result = result && counted == reportedHost[idx];
            }
            result = result && threadsInBlock == stride;
        }
        return result;
    }
} // namespace

TEMPLATE_LIST_TEST_CASE(
    "warp extents for 1D blocks smaller than the warp size",
    "[warp][warpExtents]",
    WarpTestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    // one warp per block, each warp has less lanes than the warp size
    CHECK(runLinearExtentsTest<LinearWarpExtentsKernel>(deviceExec, Vec<uint32_t, 1u>{34u}, Vec<uint32_t, 1u>{31u}));
}

TEMPLATE_LIST_TEST_CASE(
    "warp extents for a 1D block with a partial last warp",
    "[warp][warpExtents]",
    WarpTestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto const device = test::getDevice(deviceExec);
    auto const warpSize = device.getDeviceProperties().warpSize;
    // the last warp contains a single thread
    CHECK(
        runLinearExtentsTest<LinearWarpExtentsKernel>(
            deviceExec,
            Vec<uint32_t, 1u>{3u},
            Vec<uint32_t, 1u>{2u * warpSize + 1u}));
}

TEMPLATE_LIST_TEST_CASE("warp extents for full warps in a 1D block", "[warp][warpExtents]", WarpTestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto const device = test::getDevice(deviceExec);
    auto const warpSize = device.getDeviceProperties().warpSize;
    CHECK(
        runLinearExtentsTest<LinearWarpExtentsKernel>(
            deviceExec,
            Vec<uint32_t, 1u>{5u},
            Vec<uint32_t, 1u>{2u * warpSize}));
}

TEMPLATE_LIST_TEST_CASE(
    "warp extents for 1D block sizes around typical warp sizes",
    "[warp][warpExtents]",
    WarpTestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    // one value below and above common warp sizes
    for(uint32_t threads : {15u, 17u, 31u, 33u, 63u, 65u})
        CHECK(
            runLinearExtentsTest<LinearWarpExtentsKernel>(
                deviceExec,
                Vec<uint32_t, 1u>{3u},
                Vec<uint32_t, 1u>{threads}));
}

TEMPLATE_LIST_TEST_CASE("warp extents in multidimensional blocks", "[warp][warpExtents]", WarpTestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    CHECK(runMdExtentsTest(deviceExec, Vec{4u, 2u}));
    CHECK(runMdExtentsTest(deviceExec, Vec{7u, 3u}));
    CHECK(runMdExtentsTest(deviceExec, Vec{4u, 16u}));
    CHECK(runMdExtentsTest(deviceExec, Vec{4u, 32u}));
    // last dimension one value below and above common warp sizes
    CHECK(runMdExtentsTest(deviceExec, Vec{4u, 15u}));
    CHECK(runMdExtentsTest(deviceExec, Vec{4u, 17u}));
    CHECK(runMdExtentsTest(deviceExec, Vec{4u, 31u}));
    CHECK(runMdExtentsTest(deviceExec, Vec{4u, 33u}));
    CHECK(runMdExtentsTest(deviceExec, Vec{4u, 63u}));
    CHECK(runMdExtentsTest(deviceExec, Vec{4u, 65u}));
    CHECK(runMdExtentsTest(deviceExec, Vec{2u, 2u, 8u}));
}
