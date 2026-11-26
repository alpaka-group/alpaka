/* Copyright 2023 Axel Hübl, Benjamin Worpitz, Bernhard Manfred Gruber, Jan Stephan, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace alpaka;

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

struct IotaKernelND
{
    ALPAKA_FN_ACC void operator()(auto const& acc, auto out, auto outSize) const
    {
        constexpr uint32_t warpSize = onAcc::warp::getSize<ALPAKA_TYPEOF(acc)>();
        concepts::Vector auto numThreadsPerBlock = acc.getExtentsOf(onAcc::origin::block, onAcc::unit::threads);
        // test using some of the predefined warp worker and range types
        for(auto bOffset : onAcc::makeIdxMap(
                acc,
                onAcc::worker::linearBlocksInGrid,
                IdxRange{Vec{0u}, outSize, Vec{numThreadsPerBlock.product()}}))
            for(auto w : onAcc::makeIdxMap(acc, onAcc::worker::linearWarpsInBlock, onAcc::range::linearWarpsInBlock))
            {
                for(auto t :
                    onAcc::makeIdxMap(acc, onAcc::worker::linearThreadsInWarp, onAcc::range::linearThreadsInWarp))
                {
                    auto idx = bOffset + warpSize * w + t;
                    /* use atomic to avoid that multiple threads change the same location, if so we will detect it in
                     * our verification
                     */
                    if(idx < outSize)
                        onAcc::atomicAdd(acc, &out[idx].x(), idx.x());
                }
            }
    }
};

TEMPLATE_LIST_TEST_CASE("warp iota1D", "", TestApis)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);

    auto deviceProperties = device.getDeviceProperties();
    uint32_t warpSize = deviceProperties.warpSize;

    onHost::Queue queue = device.makeQueue();
    Vec extent = Vec{warpSize * 123u};
    auto dBuff = onHost::alloc<Vec<uint32_t, 1u>>(device, extent);

    auto hBuff = onHost::allocHostLike(dBuff);

    onHost::wait(queue);
    /* Use an odd number that to avoid any assumptions about number of threads.
     * Note that for a thread serialized executor alpaka will internally use a single thread per block what is equal to
     * the warp size.
     */
    auto frameSize = Vec{warpSize + 3u};
    onHost::fill(queue, dBuff, Vec{0u});
    queue.enqueue(onHost::FrameSpec{extent / frameSize, frameSize, exec}, KernelBundle{IotaKernelND{}, dBuff, extent});
    onHost::memcpy(queue, hBuff, dBuff);
    onHost::wait(queue);
    meta::ndLoopIncIdx(extent, [&](auto idx) { CHECK(idx == hBuff[idx]); });

    // MD blocks of full warps
    onHost::fill(queue, dBuff, Vec{0u});
    queue.enqueue(
        onHost::FrameSpec{Vec{5u, 3u}, Vec{3u, warpSize}, exec},
        KernelBundle{IotaKernelND{}, dBuff, extent});
    onHost::memcpy(queue, hBuff, dBuff);
    onHost::wait(queue);
    meta::ndLoopIncIdx(extent, [&](auto idx) { CHECK(idx == hBuff[idx]); });
}
