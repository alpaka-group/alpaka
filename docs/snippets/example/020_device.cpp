/* Copyright 2025 René Widera
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <iostream>

using namespace alpaka;

TEST_CASE("show host devices", "[docs]")
{
    // BEGIN-TUTORIAL-devSelect
    /* Select a device, possible combinations of api+deviceKind:
     * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
     * oneApi+amdGpu, oneApi+nvidiaGpu
     */
    auto computeDevSelector = api::host + deviceKind::cpu;
    // END-TUTORIAL-devSelect

    // BEGIN-TUTORIAL-devCount
    auto numComputeDevs = computeDevSelector.getDeviceCount();
    std::cout << "Found " << numComputeDevs << " device(s):\n";
    // END-TUTORIAL-devCount

    // BEGIN-TUTORIAL-devHandleCount
    // Always check the number of available compute devices! alpaka always creates a valid DeviceSelector even for
    // unsupported combinations of an api and deviceKind.
    if(numComputeDevs > 0)
    {
        // select the first device and get the name
        onHost::Device computeDev = computeDevSelector.makeDevice(0);
        std::cout << computeDev.getName() << "\n";
    }
    // END-TUTORIAL-devHandleCount

    // BEGIN-TUTORIAL-devProperties
    // get the device properties for each device without allocating the device
    for(auto i = 0u; i < numComputeDevs; ++i)
    {
        std::cout << "Device " << i << ":\n";
        std::cout << "  - name              " << computeDevSelector.getDeviceProperties(i).getName() << "\n";
        std::cout << "  - #multi-processors " << computeDevSelector.getDeviceProperties(i).multiProcessorCount << "\n";
        std::cout << "  - warp size         " << computeDevSelector.getDeviceProperties(i).warpSize << "\n";
    }
    // END-TUTORIAL-devProperties
}

TEST_CASE("host device", "[docs]")
{
    // BEGIN-TUTORIAL-devHostDev
    // Get a device to perform work on the host.
    // It is a shortcut compared to using the makeDeviceSelector(...) to get a host device.
    onHost::concepts::Device auto hostDevice = onHost::makeHostDevice();
    // END-TUTORIAL-devHostDev

    // Getting a queue to enqueue asynchronous work
    onHost::Queue hostQueue = hostDevice.makeQueue();
    hostQueue.enqueueHostFn([]() { std::cout << "Hallo host task" << '\n'; });
    onHost::wait(hostQueue);
}
