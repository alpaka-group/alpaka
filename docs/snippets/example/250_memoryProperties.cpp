/* Copyright 2026 SiPearl
 * SPDX-License-Identifier: MPL-2.0
 */

#include "docsTest.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <iostream>
#include <type_traits>
#include <vector>

using namespace alpaka;

TEST_CASE("memory allocations", "[docs]")
{
    onHost::concepts::Device auto device = onHost::makeHostDevice();
    {
        // BEGIN-TUTORIAL-allocBufferDev
        concepts::Vector auto extents = Vec{2u, 3u};
        // hasProperty() picks the first preference this device implements, so the line stays portable
        auto property = onHost::hasProperty(device, memoryProperty::bestLatency, memoryProperty::defaultProperty);
        // the allocation is providing a shared buffer which will be
        // automatically freed if the last handle runs out of a life-time
        concepts::IBuffer auto devBuffer = onHost::alloc<int>(device, extents, property);
        // END-TUTORIAL-allocBufferDev
        unused(devBuffer);
    }
    {
        // BEGIN-TUTORIAL-allocBufferMapped
        concepts::Vector auto extents = Vec{2u, 3u};
        auto property = onHost::hasProperty(device, memoryProperty::bestBandwidth, memoryProperty::defaultProperty);
        // allocate memory which lives on the host but is accessible from the compute device too
        concepts::IBuffer auto devMappedBuffer = onHost::allocMapped<int>(device, extents, property);
        // END-TUTORIAL-allocBufferMapped
        unused(devMappedBuffer);
    }
    {
        // BEGIN-TUTORIAL-allocBufferUnified
        concepts::Vector auto extents = Vec{2u, 3u};
        auto property = onHost::hasProperty(device, memoryProperty::locality, memoryProperty::defaultProperty);
        // allocate memory can be accessed from host and device (unified memory),
        // the real location depends on the native backend e.g. CUDA, OneApi, ...
        concepts::IBuffer auto devUnifiedBuffer = onHost::allocUnified<int>(device, extents, property);
        // END-TUTORIAL-allocBufferUnified
        unused(devUnifiedBuffer);
    }
}

void callKernel([[maybe_unused]] auto dummyMemory)
{
}

TEST_CASE("memory allocations deferred", "[docs]")
{
    onHost::concepts::Device auto device = onHost::makeHostDevice();
    onHost::Queue queue = device.makeQueue();
    // BEGIN-TUTORIAL-allocBufferDeferred
    concepts::Vector auto extents = Vec{2u, 3u};
    // a queue resolves to the memory properties of its device
    auto property = onHost::hasProperty(queue, memoryProperty::bestBandwidth, memoryProperty::defaultProperty);
    {
        // The allocation is deferred.
        // It is only allowed to access the memory after the queue processed the allocation task.
        concepts::IBuffer auto devDeferredBuffer = onHost::allocDeferred<int>(queue, extents, property);
        // Call the kernel with the buffer. This is only a dummy call not a real kernel call.
        callKernel(devDeferredBuffer);
        // At the end of the scope the buffer will be destroyed, but it could be that the kernel is not finished yet.
        // This special allocation method take care that the buffer is waiting for the queue before the memory is
        // freed.
    }
    // END-TUTORIAL-allocBufferDeferred
}

TEST_CASE("memory allocations portable", "[docs]")
{
    onHost::concepts::Device auto device = onHost::makeHostDevice();
    // BEGIN-TUTORIAL-allocPortable
    concepts::Vector auto extents = Vec{2u, 3u};
    // hasProperty() scans the preference list from left to right and returns the first property this device
    // implements, therefore the same line compiles for every backend
    concepts::IBuffer auto buffer = onHost::alloc<int>(
        device,
        extents,
        onHost::hasProperty(
            device,
            memoryProperty::bestBandwidth,
            memoryProperty::locality,
            memoryProperty::defaultProperty));
    // query the support explicitly if a different algorithm should be used
    static_assert(onHost::isMemoryPropertySupportedBy(memoryProperty::defaultProperty, device));
    // END-TUTORIAL-allocPortable
    unused(buffer);
}

TEST_CASE("memory allocations like", "[docs]")
{
    onHost::concepts::Device auto computeDevice = onHost::makeHostDevice();
    // BEGIN-TUTORIAL-allocLike
    concepts::Vector auto extents = Vec{2u, 3u};
    // allocHost() takes no device argument, it always allocates on the host CPU device, so query that one
    auto hostDevice = onHost::makeHostDevice();
    // short notation to allocate memory on the host without a host device as first argument
    concepts::IBuffer auto hostBuffer = onHost::allocHost<int>(
        extents,
        onHost::hasProperty(hostDevice, memoryProperty::bestLatency, memoryProperty::defaultProperty));
    // inherits value type and extents but does NOT copy the data
    concepts::IBuffer auto devDoubleBuffer = onHost::allocLike(
        computeDevice,
        hostBuffer,
        onHost::hasProperty(computeDevice, memoryProperty::bestBandwidth, memoryProperty::defaultProperty));
    // END-TUTORIAL-allocLike

    unused(devDoubleBuffer);
}
