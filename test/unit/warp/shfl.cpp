/* Copyright 2025 Sergei Bastrakov, Bernhard Manfred Gruber, Jan Stephan, Andrea Bocci, Aurora Perego, Mehmet
 * Yusufoglu, René Widera SPDX-License-Identifier: MPL-2.0
 */

/** @file Tests the warp "shfl" (shuffle) operation which broadcasts a value from one lane to all lanes.
 * The "shfl" warp operation allows each thread to read a value from a specified source lane's register.
 * It's a data exchange operation that enables direct thread-to-thread communication within a warp without shared
 * memory.
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
    struct ShflMultiThreadKernel
    {
        template<typename T_Acc>
        ALPAKA_FN_ACC void operator()(T_Acc const& acc, concepts::IMdSpan<bool> auto success) const
        {
            constexpr uint32_t warpExtent = onAcc::warp::getSize<ALPAKA_TYPEOF(acc)>();

            // number of threads should be a multiple of the warp size
            warpCheck(success, warpExtent >= 1u);

            // Lane ID drives the expected source values for each shuffle check.
            uint32_t const lane = onAcc::warp::getLaneIdx(acc);

            // Exercise trivial zero-offset and max-offset cases.
            // Broadcasting from literal lane 0 must work regardless of the caller lane.
            warpCheck(success, onAcc::warp::shfl(acc, 42, 0u) == 42);
            // Using the current lane as the payload and requesting src=0 should always give back 0.
            warpCheck(success, onAcc::warp::shfl(acc, lane, 0u) == 0);
            if constexpr(warpExtent >= 2)
            {
                // Requesting src=1 broadcasts lane 1's value to every participant.
                // test requires at least two threads in a warp
                warpCheck(success, onAcc::warp::shfl(acc, lane, 1u) == 1);
            }

            // Large src index is clamped to the logical width; value must remain unchanged.
            warpCheck(success, onAcc::warp::shfl(acc, 5, std::numeric_limits<uint32_t>::max()) == 5);

            auto const epsilon = std::numeric_limits<float>::epsilon();
            for(uint32_t width = 1; width < warpExtent; width *= 2)
            {
                // Check every logical partition width supported by the backend.
                for(uint32_t idx = 0; idx < width; ++idx)
                {
                    auto const section = width * (lane / width);
                    // Integer payloads should resolve to the subgroup-relative source index.
                    auto const shuffle = onAcc::warp::shfl(acc, lane, idx, width);
                    warpCheck(success, shuffle == idx + section);

                    // Floating payloads exercise non-integral types under the same subgroup restriction.
                    auto const ans = onAcc::warp::shfl(acc, 4.0f - static_cast<float>(lane), idx, width);
                    auto const expect = 4.0f - static_cast<float>(idx + section);
                    warpCheck(success, alpaka::math::abs(ans - expect) < epsilon);
                }
            }

            if(static_cast<int>(lane) >= static_cast<int>(warpExtent / 2u))
            {
                warpCheck(success, onAcc::warp::shfl(acc, 42, warpExtent - 1u) == 42);
                // Upper half should be fully masked from the final checks.
                return;
            }
            else
            {
                // check that shfl can be called within branches of the same level
                warpCheck(success, onAcc::warp::shfl(acc, 11, 0u) == 11);
            }
            // int is used to silence cast warning because warpExtent can be zero during the host path evaluation
            for(int idxTmp = 0u; idxTmp < static_cast<int>(warpExtent) / 2; ++idxTmp)
            {
                uint32_t idx = static_cast<uint32_t>(idxTmp);
                // Active sub-group must always read the value produced by the chosen lane.
                // Within the lower half, shuffling with src=idx must reproduce the selected lane.
                warpCheck(success, onAcc::warp::shfl(acc, lane, idx) == idx);
                auto const ans = onAcc::warp::shfl(acc, 4.0f - static_cast<float>(lane), idx);
                // Float payload confirms the same behaviour holds across types for the masked subgroup.
                auto const expect = 4.0f - static_cast<float>(idx);
                warpCheck(success, alpaka::math::abs(ans - expect) < epsilon);
            }
        }
    };
} // namespace

TEMPLATE_LIST_TEST_CASE("warp shfl moves values between lanes", "[warp][shfl]", WarpTestBackends)
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
    queue.enqueue(onHost::FrameSpec{blocks, threads, exec}, KernelBundle{ShflMultiThreadKernel{}, successDev});
    onHost::memcpy(queue, successHost, successDev);
    onHost::wait(queue);
    CHECK(successHost[0]);
}
