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
        // the allocation is providing a shared buffer which will be
        // automatically freed if the last handle runs out of a life-time
        concepts::IBuffer auto devBuffer = onHost::alloc<int>(device, extents, memoryProperty::bestLatency);
        // END-TUTORIAL-allocBufferDev
        unused(devBuffer);
    }
    {
        // BEGIN-TUTORIAL-allocBufferMapped
        concepts::Vector auto extents = Vec{2u, 3u};
        // allocate memory which lives on the host but is accessible from the compute device too
        concepts::IBuffer auto devMappedBuffer
            = onHost::allocMapped<int>(device, extents, memoryProperty::bestBandwidth);
        // END-TUTORIAL-allocBufferMapped
        unused(devMappedBuffer);
    }
    {
        // BEGIN-TUTORIAL-allocBufferUnified
        concepts::Vector auto extents = Vec{2u, 3u};
        // allocate memory can be accessed from host and device (unified memory),
        // the real location depends on the native backend e.g. CUDA, OneApi, ...
        concepts::IBuffer auto devUnifiedBuffer = onHost::allocUnified<int>(device, extents, memoryProperty::locality);
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
    {
        // The allocation is deferred.
        // It is only allowed to access the memory after the queue processed the allocation task.
        concepts::IBuffer auto devDeferredBuffer
            = onHost::allocDeferred<int>(queue, extents, memoryProperty::bestBandwidth);
        // Call the kernel with the buffer. This is only a dummy call not a real kernel call.
        callKernel(devDeferredBuffer);
        // At the end of the scope the buffer will be destroyed, but it could be that the kernel is not finished yet.
        // This special allocation method take care that the buffer is waiting for the queue before the memory is
        // freed.
    }
    // END-TUTORIAL-allocBufferDeferred
}

TEST_CASE("memory allocations like", "[docs]")
{
    onHost::concepts::Device auto computeDevice = onHost::makeHostDevice();
    // BEGIN-TUTORIAL-allocLike
    concepts::Vector auto extents = Vec{2u, 3u};
    // short notation to allocate memory on the host without a host device as first argument
    concepts::IBuffer auto hostBuffer = onHost::allocHost<int>(extents, memoryProperty::bestLatency);
    // inherits value type and extents but does NOT copy the data
    concepts::IBuffer auto devDoubleBuffer
        = onHost::allocLike(computeDevice, hostBuffer, memoryProperty::bestBandwidth);
    // END-TUTORIAL-allocLike

    unused(devDoubleBuffer);
}
