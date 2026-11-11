/* Copyright 2025 René Widera
 * SPDX-License-Identifier: ISC
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

TEST_CASE("memory extent forms", "[docs]")
{
    // BEGIN-TUTORIAL-allocBufferExtentForms
    auto oneDimBuffer = onHost::allocHost<int>(size_t{16});

    auto twoDimExtents = Vec<uint32_t, 2u>{2u, 3u};
    auto twoDimBuffer = onHost::allocHost<int>(twoDimExtents);

    static_assert(std::same_as<ALPAKA_TYPEOF(twoDimBuffer)::index_type, uint32_t>);
    // END-TUTORIAL-allocBufferExtentForms

    CHECK(oneDimBuffer.getExtents().x() == 16u);
    CHECK(twoDimBuffer.getExtents() == twoDimExtents);
}

TEST_CASE("memory allocations", "[docs]")
{
    onHost::concepts::Device auto device = onHost::makeHostDevice();
    {
        // BEGIN-TUTORIAL-allocBufferDev
        concepts::Vector auto extents = Vec{2u, 3u};
        // the allocation is providing a shared buffer which will be
        // automatically freed if the last handle runs out of a life-time
        concepts::IBuffer auto devBuffer = onHost::alloc<int>(device, extents);
        // END-TUTORIAL-allocBufferDev
        alpaka::unused(devBuffer);
    }
    {
        // BEGIN-TUTORIAL-allocBufferMapped
        concepts::Vector auto extents = Vec{2u, 3u};
        // allocate memory which lives on the host but is accessible from the compute device too
        concepts::IBuffer auto devMappedBuffer = onHost::allocMapped<int>(device, extents);
        // END-TUTORIAL-allocBufferMapped
        alpaka::unused(devMappedBuffer);
    }
    {
        // BEGIN-TUTORIAL-allocBufferUnified
        concepts::Vector auto extents = Vec{2u, 3u};
        // allocate memory can be accessed from host and device (unified memory),
        // the real location depends on the native backend e.g. CUDA, OneApi, ...
        concepts::IBuffer auto devUnifiedBuffer = onHost::allocUnified<int>(device, extents);
        // END-TUTORIAL-allocBufferUnified
        alpaka::unused(devUnifiedBuffer);
    }
}

// mimics a kernel call, which takes its arguments by value
// NOLINTNEXTLINE(performance-unnecessary-value-param)
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
        concepts::IBuffer auto devDeferredBuffer = onHost::allocDeferred<int>(queue, extents);
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
    concepts::IBuffer auto hostBuffer = onHost::allocHost<int>(extents);
    // inherits value type and extents but does NOT copy the data
    concepts::IBuffer auto devDoubleBuffer = onHost::allocLike(computeDevice, hostBuffer);
    // END-TUTORIAL-allocLike

    alpaka::unused(devDoubleBuffer);
}
