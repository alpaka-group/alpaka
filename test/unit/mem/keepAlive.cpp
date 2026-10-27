/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <future>
using namespace alpaka;

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;
using IdxType = std::size_t;
using ValType = int;

struct IotaKernel
{
    ALPAKA_FN_ACC void operator()(auto const& acc, concepts::IMdSpan<ValType> auto out) const
    {
        // the kernel should run for some time to increase the likelihood of a race condition occurring and the test
        // failing in case the keepAlive isn't working
        // proper synchronization isn't really possible generically and well-behavedly
        for(auto i = 0; i < 1000; ++i)
        {
            for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange(out.getExtents())))
                onAcc::atomicExch(acc, &out[i], static_cast<ValType>(i.x()));
        }
    }
};

TEMPLATE_LIST_TEST_CASE("keep alive", "", TestApis)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);
    onHost::Queue queue = device.makeQueue();

    constexpr std::size_t n = 256;
    auto originalBuffer = onHost::alloc<int>(device, n);

    {
        // enqueue everything in this scope
        auto scopedBuffer = onHost::allocDeferred<int>(queue, n);

        auto framespec = getFrameSpec(device, exec, scopedBuffer.getExtents());

        // fill scoped buffer
        queue.enqueue(framespec, KernelBundle{IotaKernel{}, scopedBuffer});

        // copy back to the original
        onHost::memcpy(queue, originalBuffer, scopedBuffer);

        // scoped buffer runs out of scope, keep alive until the queue gets here
        scopedBuffer.keepAlive(queue);
    }
    onHost::wait(queue);

    // validate that the original buffer got the iota correctly copied
    auto hostResult = onHost::allocHostLike(originalBuffer);

    onHost::memcpy(queue, hostResult, originalBuffer);
    onHost::wait(queue);

    bool correct = true;
    for(IdxType i = 0; i < n; ++i)
    {
        if(hostResult[i] != static_cast<ValType>(i))
            correct = false;
    }

    REQUIRE(correct);
}
