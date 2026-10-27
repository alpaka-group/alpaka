/* Copyright 2025 Luca Venerando Greco, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <vector>

using namespace alpaka;

using TestBackends = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

struct PopcountKernel
{
    template<typename T_Acc>
    ALPAKA_FN_ACC void operator()(T_Acc const& acc, concepts::IMdSpan auto output, concepts::IMdSpan auto const input)
        const
    {
        for(auto [index] : onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, IdxRange{input.getExtents()}))
        {
            output[index] = clz(input[index]);
        }
    }
};

TEMPLATE_LIST_TEST_CASE("clz", "[intrinsic][clz]", TestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device devAcc = test::getDevice(deviceExec);
    concepts::Executor auto computeExec = test::getExecutor(deviceExec);

    // Create a queue on the device
    alpaka::onHost::Queue queue = devAcc.makeQueue();

    // Input data
    std::vector<uint64_t> hostInput = {0, 1, 2, 1llu << 32, 1llu << 63};
    size_t const size = hostInput.size();

    // Allocate device memory
    auto devInput = alpaka::onHost::alloc<uint64_t>(devAcc, hostInput.size());
    auto devOutput = alpaka::onHost::alloc<int>(devAcc, hostInput.size());

    // Copy data from host to device
    alpaka::onHost::memcpy(queue, devInput, hostInput);

    // Define execution parameters
    auto const frameSpec = alpaka::onHost::getFrameSpec(devAcc, computeExec, devInput.getExtents());

    // Create kernel
    PopcountKernel kernel;
    auto const taskKernel = alpaka::KernelBundle{kernel, devOutput, devInput};

    // Execute the kernel
    queue.enqueue(frameSpec, taskKernel);

    // Copy data from device to host
    std::vector<int> hostOutput(size);
    alpaka::onHost::memcpy(queue, hostOutput, devOutput);

    // Wait for the queue to finish
    alpaka::onHost::wait(queue);

    // Verification
    for(size_t i = 0; i < size; ++i)
    {
        auto val = hostInput[i];
        int expected = std::countl_zero(val);
        CHECK(hostOutput[i] == expected);
    }
}
