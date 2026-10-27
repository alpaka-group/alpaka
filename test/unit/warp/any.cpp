/* Copyright 2025 Sergei Bastrakov, Bernhard Manfred Gruber, Jan Stephan, Andrea Bocci, Aurora Perego, Mehmet
 * Yusufoglu, René Widera SPDX-License-Identifier: MPL-2.0
 */

/** @file Tests the warp "any" operation which checks if at least one lane in the warp satisfies a predicate.
 * The "any" warp operation returns true if any thread in the warp evaluates the condition as true.
 * It's a collective operation that gathers votes across all participating threads executing in lockstep.
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
    struct AnyMultiThreadKernel
    {
        template<typename T_Acc>
        ALPAKA_FN_ACC void operator()(T_Acc const& acc, concepts::IMdSpan<bool> auto success, std::uint32_t idx) const
        {
            // test if the warp size can be constexpr
            constexpr uint32_t warpExtent = onAcc::warp::getSize<ALPAKA_TYPEOF(acc)>();
            /* We can not use a static_assert for testing because the compiler will evaluate the warp size during the
             * host parsing to what will result in false negatives */
            warpCheck(success, warpExtent >= 1u);

            auto const threadsPerBlock = static_cast<std::int32_t>(acc[alpaka::layer::thread].count().product());
            // number of threads should be a multiple of the warp size
            warpCheck(success, threadsPerBlock % warpExtent == 0);

            auto const lane = static_cast<std::int32_t>(onAcc::warp::getLaneIdx(acc));
            if(lane % 2 != 0)
            {
                // Only even lanes participate; others exit silently.
                return;
            }

            // No participating lane votes true, so the result must stay false.
            warpCheck(success, !onAcc::warp::any(acc, 0));
            // At least one active lane votes true, toggling the aggregate to true.
            warpCheck(success, onAcc::warp::any(acc, 42));

            auto const castIdx = static_cast<std::int32_t>(idx);

            if constexpr(warpExtent >= 2)
            {
                // Only the non-targeted even lanes vote true here; the chosen lane contributes false, so the
                // collective stays true. Example: active lanes {0,2,4,6}; choosing idx=2 yields predicates {1,0,1,1}.
                warpCheck(success, onAcc::warp::any(acc, lane == castIdx ? 0 : 1));
            }

            auto const expected = (idx % 2u == 0u);
            // The inverse predicate now gives the selected lane the only true vote; the result is true only if that
            // lane is active. Example: active lanes {0,2,4,6}; choosing idx=2 yields predicates {0,1,0,0}, so the vote
            // is true.
            warpCheck(success, onAcc::warp::any(acc, lane == castIdx ? 1 : 0) == expected);
        }
    };
} // namespace

TEMPLATE_LIST_TEST_CASE("warp any vote observes active lanes", "[warp][any]", WarpTestBackends)
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
        // Rotate through each candidate lane to ensure mask-gated votes succeed.
        onHost::memset(queue, successDev, static_cast<std::uint8_t>(true));
        queue.enqueue(frame, KernelBundle{AnyMultiThreadKernel{}, successDev, idx});
        onHost::memcpy(queue, successHost, successDev);
        onHost::wait(queue);
        INFO("idx=" << idx);
        CHECK(successHost[0]);
    }
}
