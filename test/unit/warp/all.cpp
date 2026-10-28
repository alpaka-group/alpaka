/* Copyright 2025 Sergei Bastrakov, Bernhard Manfred Gruber, Jan Stephan, Andrea Bocci, Aurora Perego, Mehmet
 * Yusufoglu, René Widera SPDX-License-Identifier: MPL-2.0
 */

/** @file Tests the warp "all" operation which checks if all lanes in the warp satisfy a predicate.
 * The "all" warp operation returns true only when every participating thread evaluates the condition as true.
 * It's a collective operation that requires unanimous agreement across all active threads in the warp.
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
    struct AllMultiThreadKernel
    {
        template<typename T_Acc>
        ALPAKA_FN_ACC void operator()(T_Acc const& acc, concepts::IMdSpan<bool> auto success, std::uint32_t idx) const
        {
            constexpr uint32_t warpExtent = onAcc::warp::getSize<ALPAKA_TYPEOF(acc)>();

            auto const threadsPerBlock = static_cast<std::int32_t>(acc[alpaka::layer::thread].count().product());
            // number of threads should be a multiple of the warp size
            warpCheck(success, threadsPerBlock % warpExtent == 0);

            auto const lane = static_cast<std::int32_t>(onAcc::warp::getLaneIdx(acc));
            if(lane % 3 != 0)
            {
                // Only every third lane participates in the collective vote.
                // Other lanes exit silently below.
                return;
            }

            // All participating lanes vote false hence the inverse must be true.
            warpCheck(success, !onAcc::warp::all(acc, 0));

            // assumes non-zero values evaluate as true
            // All participating lanes vote true hence the result must be true.
            warpCheck(success, onAcc::warp::all(acc, 42));


            auto const castIdx = static_cast<std::int32_t>(idx);

            // requires at least two threads
            if constexpr(warpExtent >= 2)
            {
                // Example: active lanes {0,3,6}; choosing idx=3 yields predicates {0,1,0}, so unanimity fails.
                warpCheck(success, !onAcc::warp::all(acc, lane == castIdx ? 1 : 0));
            }

            auto const expected = (idx % 3u != 0u);
            // Every active lane except the triggering one votes true; the result is true only if that lane is
            // inactive. Example: idx=4 (masked lane) leaves predicates {1,1,1} and the vote succeeds, whereas idx=3
            // produces {1,0,1} and fails.
            warpCheck(success, onAcc::warp::all(acc, lane == castIdx ? 0 : 1) == expected);
        }
    };
} // namespace

TEMPLATE_LIST_TEST_CASE("warp all vote honours only active lanes", "[warp][all]", WarpTestBackends)
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
    auto const threads = Vec<std::uint32_t, 1u>{4u * warpExtent};
    auto const frame = onHost::FrameSpec{blocks, threads, exec};

    for(std::uint32_t idx = 0u; idx < warpExtent; ++idx)
    {
        // Iterate over each potential trigger lane to verify masked votes.
        onHost::memset(queue, successDev, static_cast<std::uint8_t>(true));
        queue.enqueue(frame, KernelBundle{AllMultiThreadKernel{}, successDev, idx});
        onHost::memcpy(queue, successHost, successDev);
        onHost::wait(queue);
        INFO("idx=" << idx);
        CHECK(successHost[0]);
    }
}
