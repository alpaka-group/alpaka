/* Copyright 2024 Andrea Bocci, René Widera
 * SPDX-License-Identifier: Apache-2.0
 */

#include "config.h"

#include <alpaka/alpaka.hpp>

#include <cstdlib>
#include <iostream>
#include <vector>

int example(auto const devSpec)
{
    // initialise the accelerator platform
    auto devSelector = alpaka::onHost::makeDeviceSelector(devSpec);

    // require at least one device
    std::size_t n = devSelector.getDeviceCount();

    if(n == 0)
    {
        return EXIT_FAILURE;
    }

    // use a std::vector as host buffer
    uint32_t size = 42;
    std::vector<float> host_data(size);
    std::cout << "host vector at " << std::data(host_data) << "\n\n";

    // fill the host buffers with values
    for(uint32_t i = 0; i < size; ++i)
    {
        host_data[i] = i;
    }

    // use the first device
    alpaka::onHost::Device device = devSelector.makeDevice(0);
    std::cout << "Device: " << alpaka::onHost::getName(device) << '\n';

    // create a work queue
    alpaka::onHost::Queue queue = device.makeQueue();
    {
        // allocate a buffer of floats in global device memory
        auto device_buffer = alpaka::onHost::alloc<float>(device, size);
        std::cout << "memory buffer on " << alpaka::onHost::getStaticName(alpaka::getApi(device_buffer)) << " at "
                  << std::data(device_buffer) << "\n\n";

        // once the device buffer goes out of scope, it will enqueue its free, so all other operations have finished
        // before the memory is freed
        device_buffer.destructorWaitFor(queue);

        // set the device memory to all zeros (byte-wise, not element-wise)
        alpaka::onHost::memset(queue, device_buffer, 0x00);

        // create a view to the device data
        auto view = alpaka::View(device_buffer);

        // copy the contents of the device buffer to the host buffer
        alpaka::onHost::memcpy(queue, host_data, view);
    }

    // wait for all operations to complete
    alpaka::onHost::wait(queue);

    // read the content of the host buffer
    for(uint32_t i = 0; i < size; ++i)
    {
        std::cout << host_data[i] << ' ';
    }
    std::cout << '\n';

    std::cout << "All work has completed\n";

    return EXIT_SUCCESS;
}

auto main() -> int
{
    using namespace alpaka;

    /* Execute the example once for each device specification
     *
     * If you would like to execute it for a single device only you can use the following code.
     *  @code{.cpp}
     *  auto deviceSpec = onHost::DeviceSpec{api::cuda, deviceKind::nvidiaGpu};
     *  return example(deviceSpec);
     *  @endcode
     *
     * Some examples for device specifications (depending on the active dependencies).
     *
     *   onHost::DeviceSpec{api::host, deviceKind::cpu}
     *   onHost::DeviceSpec{api::cuda, deviceKind::nvidiaGpu}
     *   onHost::DeviceSpec{api::hip, deviceKind::amdGpu}
     *   onHost::DeviceSpec{api::oneApi, deviceKind::intelGpu}
     *
     * A list of api's and device kinds can be found
     * https://alpaka.readthedocs.io/en/latest/basic/cheatsheet.html#available-apis
     */
    return onHost::executeForEachIfHasDevice(
        [=](auto const& devSpec) { return example(devSpec); },
        onHost::getDeviceSpecsFor(onHost::enabledApis));
}
