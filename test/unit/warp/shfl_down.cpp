/* Copyright 2025 Sergei Bastrakov, Bernhard Manfred Gruber, Jan Stephan, Andrea Bocci, Aurora Perego, Mehmet
 * Yusufoglu, René Widera SPDX-License-Identifier: MPL-2.0
 */

/** @file Tests the warp "shfl_down" (shuffle down) operation which shifts values toward higher-numbered lanes.
 * The "shfl_down" warp operation allows each thread to read a value from a lane at a fixed offset below.
 * It's a data exchange operation useful for prefix scans and reduction patterns within a warp.
 */

#include "utils.hpp"

#include <alpaka/onAcc/warp.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>

using namespace alpaka;
using alpaka::test::warp::warpCheck;
using alpaka::test::warp::WarpTestBackends;

namespace
{
    struct ShflDownMultiThreadKernel
    {
        template<typename T_Acc>
        ALPAKA_FN_ACC void operator()(T_Acc const& acc, concepts::IMdSpan<bool> auto success) const
        {
            auto const warpExtent = static_cast<std::int32_t>(onAcc::warp::getSize(acc));
            warpCheck(success, warpExtent >= 1);

            auto const threadsPerBlock = static_cast<std::int32_t>(acc[alpaka::layer::thread].count().product());
            warpCheck(success, threadsPerBlock % warpExtent == 0);

            auto const lane = static_cast<std::int32_t>(onAcc::warp::getLaneIdx(acc));

            // With zero offset, every lane should see the source literal unchanged.
            warpCheck(success, onAcc::warp::shflDown(acc, 42, 0u) == 42);
            // A zero-width shuffle leaves each lane's original value intact.
            warpCheck(success, onAcc::warp::shflDown(acc, lane, 0u) == lane);
            // Offset of one shifts values toward higher indices, clamping at the partition boundary.
            // For example, lane 0 with offset 1 should see lane 1's value, lane 1 should see lane 2's value, and so
            // on, with the last lane seeing its own value.
            warpCheck(success, onAcc::warp::shflDown(acc, lane, 1u) == (lane + 1 < warpExtent ? lane + 1 : lane));

            auto const epsilon = std::numeric_limits<float>::epsilon();
            for(int width = 1; width < warpExtent; width *= 2)
            {
                // Validate every partition width the backend claims to support.
                for(int idx = 0; idx < width; ++idx)
                {
                    auto const sectionStart = width * (lane / width);
                    auto const sectionEnd = sectionStart + width;
                    auto const shuffled = onAcc::warp::shflDown(
                        acc,
                        lane,
                        static_cast<std::uint32_t>(idx),
                        static_cast<std::uint32_t>(width));
                    auto const expectedInt = (lane + idx < sectionEnd) ? lane + idx : lane;
                    // Each lane should see the value from the lane at the offset below, or clamp to its own value if
                    // out of range. For example, if lane 2 with offset 1 should see lane 3's value, but if lane 3 is
                    // the last lane in the partition, it sees its own value.
                    warpCheck(success, shuffled == expectedInt);

                    auto const ans = onAcc::warp::shflDown(
                        acc,
                        4.0f - static_cast<float>(lane),
                        static_cast<std::uint32_t>(idx),
                        static_cast<std::uint32_t>(width));
                    auto const expect = (lane + idx < sectionEnd) ? 4.0f - static_cast<float>(lane + idx)
                                                                  : 4.0f - static_cast<float>(lane);
                    warpCheck(success, alpaka::math::abs(ans - expect) < epsilon);
                }
            }

            if(lane >= warpExtent / 2)
            {
                warpCheck(success, onAcc::warp::shflDown(acc, 42, 1u) == 42);
                // Mask out the upper sub-group for the final spot checks.
                return;
            }
            else
            {
                // warpCheck(success, onAcc::warp::shflDown(acc, 43, 1u) == 43);
            }

            for(int idx = 0; idx < warpExtent / 2; ++idx)
            {
                // Active lanes must march forward until the end of the logical sub-group.
                auto const shuffled = onAcc::warp::shflDown(acc, lane, static_cast<std::uint32_t>(idx));
                auto const ans
                    = onAcc::warp::shflDown(acc, 4.0f - static_cast<float>(lane), static_cast<std::uint32_t>(idx));
                auto const expectFloat = (lane + idx < warpExtent / 2) ? 4.0f - static_cast<float>(lane + idx) : 0.0f;

                if(lane + idx < warpExtent / 2)
                {
                    // Each lane should see the value from the lane at the offset below within the active sub-group.
                    // Example: lane 0 with offset 1 reads lane 1, lane 1 reads lane 2, etc., until the subgroup ends.
                    warpCheck(success, shuffled == lane + idx);
                    // Floating payload mirrors the integer expectation for the same in-range partner lane.
                    warpCheck(success, alpaka::math::abs(ans - expectFloat) < epsilon);
                }
            }
        }
    };
} // namespace

TEMPLATE_LIST_TEST_CASE("warp shflDown shifts toward higher lanes", "[warp][shfl_down]", WarpTestBackends)
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

    onHost::memset(queue, successDev, static_cast<std::uint8_t>(true));
    queue.enqueue(onHost::FrameSpec{blocks, threads, exec}, KernelBundle{ShflDownMultiThreadKernel{}, successDev});
    onHost::memcpy(queue, successHost, successDev);
    onHost::wait(queue);
    CHECK(successHost[0]);
}
