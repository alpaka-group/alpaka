/* Copyright 2025 Sergei Bastrakov, Bernhard Manfred Gruber, Jan Stephan, Andrea Bocci, Aurora Perego, Mehmet
 * Yusufoglu, René Widera SPDX-License-Identifier: MPL-2.0
 */

/** @file Tests that the warp activemask helper reports exactly the lanes participating in execution.
 * Warp operations are SIMT collectives that act on the threads executing in lockstep on a GPU warp.
 */

#include "utils.hpp"

#include <alpaka/onAcc/warp.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>

using namespace alpaka;
using alpaka::test::warp::fullMask;
using alpaka::test::warp::singleBit;
using alpaka::test::warp::warpCheck;
using alpaka::test::warp::WarpTestBackends;

namespace
{
    struct ActivemaskMultiThreadKernel
    {
        template<typename T_Acc>
        ALPAKA_FN_ACC void operator()(
            T_Acc const& acc,
            concepts::IMdSpan<bool> auto success,
            std::uint32_t inactiveLane) const
        {
            // test if the warp size can be constexpr
            constexpr uint32_t warpExtent = onAcc::warp::getSize<ALPAKA_TYPEOF(acc)>();
            /* We can not use a static_assert for testing because the compiler will evaluate the warp size during the
             * host parsing to what will result in false negatives */
            warpCheck(success, warpExtent >= 1u);

            /* We start on the host side a frame specification with a frame extent of the warp size.
             * alpaka should not reduce the number of threads to a value smaller than the warp size if the user is not
             * applying for it.
             */
            auto const threadsPerBlock = static_cast<std::uint32_t>(acc[alpaka::layer::thread].count().product());
            // number of threads should be a multiple of the warp size
            warpCheck(success, threadsPerBlock % warpExtent == 0);

            auto const lane = onAcc::warp::getLaneIdx(acc);

            if(lane == inactiveLane)
            {
                // Early exit: mark this lane inactive without touching the mask.
                return;
            }

            auto const mask = onAcc::warp::activemask(acc);

            auto const expected = fullMask(warpExtent) & ~singleBit(inactiveLane);
            warpCheck(success, mask == expected);
        }
    };
} // namespace

TEMPLATE_LIST_TEST_CASE("warp activemask reflects participating lanes", "[warp][activemask]", WarpTestBackends)
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

    for(std::uint32_t inactiveLane = 0u; inactiveLane < warpExtent; ++inactiveLane)
    {
        // Sweep every lane once to confirm the mask drops exactly that participant.
        onHost::memset(queue, successDev, static_cast<std::uint8_t>(true));
        queue.enqueue(frame, KernelBundle{ActivemaskMultiThreadKernel{}, successDev, inactiveLane});
        onHost::memcpy(queue, successHost, successDev);
        onHost::wait(queue);
        INFO("inactiveLane=" << inactiveLane);
        CHECK(successHost[0]);
    }
}
