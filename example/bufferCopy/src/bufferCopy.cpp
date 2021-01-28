/* Copyright 2019-2021 Alexander Matthes, Benjamin Worpitz, Erik Zenker, Matthias Werner, Bernhard Manfred Gruber
 *
 * This file exemplifies usage of alpaka.
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED “AS IS” AND ISC DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL ISC BE LIABLE FOR ANY
 * SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR
 * IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <cstdint>
#include <iostream>

template<size_t width>
ALPAKA_FN_ACC size_t linIdxToPitchedIdx(size_t const globalIdx, size_t const pitch)
{
    const size_t idx_x = globalIdx % width;
    const size_t idx_y = globalIdx / width;
    return idx_x + idx_y * pitch;
}

//! Prints all elements of the buffer.
struct PrintBufferKernel
{
    template<typename TAcc, typename TData>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        alpaka::experimental::BufferAccessor<TAcc, TData, 3, alpaka::experimental::ReadAccess> const data) const
        -> void
    {
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const gridSize = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        for(size_t z = idx[0]; z < data.extents[0]; z += gridSize[0])
            for(size_t y = idx[1]; y < data.extents[1]; y += gridSize[1])
                for(size_t x = idx[2]; x < data.extents[2]; x += gridSize[2])
                    printf("%zu,%zu,%zu:%u ", z, y, x, static_cast<uint32_t>(data[{z, y, x}]));
    }
};

//! Tests if the value of the buffer on index i is equal to i.
struct TestBufferKernel
{
    template<typename TAcc, typename TData>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        alpaka::experimental::BufferAccessor<TAcc, TData, 3, alpaka::experimental::ReadAccess> const data) const
        -> void
    {
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const gridSize = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        for(size_t z = idx[0]; z < data.extents[0]; z += gridSize[0])
            for(size_t y = idx[1]; y < data.extents[1]; y += gridSize[1])
                for(size_t x = idx[2]; x < data.extents[2]; x += gridSize[2])
                    ALPAKA_ASSERT_OFFLOAD(
                        data(z, y, x) == alpaka::mapIdx<1u>(decltype(data.extents){z, y, x}, data.extents)[0]);
    }
};

//! Fills values of buffer with increasing elements starting from 0
struct FillBufferKernel
{
    template<typename TAcc, typename TData>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        alpaka::experimental::BufferAccessor<TAcc, TData, 3, alpaka::experimental::WriteAccess> const data) const
        -> void
    {
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const gridSize = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        for(size_t z = idx[0]; z < data.extents[0]; z += gridSize[0])
            for(size_t y = idx[1]; y < data.extents[1]; y += gridSize[1])
                for(size_t x = idx[2]; x < data.extents[2]; x += gridSize[2])
                    data(z, y, x) = alpaka::mapIdx<1u>(idx, data.extents)[0];
    }
};

auto main() -> int
{
// Fallback for the CI with disabled sequential backend
#if defined(ALPAKA_CI) && !defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
    return EXIT_SUCCESS;
#else
    // Define the index domain
    using Dim = alpaka::DimInt<3u>;
    using Idx = std::size_t;

    // Define the device accelerator
    //
    // It is possible to choose from a set of accelerators:
    // - AccGpuCudaRt
    // - AccGpuHipRt
    // - AccCpuThreads
    // - AccCpuFibers
    // - AccCpuOmp2Threads
    // - AccCpuOmp2Blocks
    // - AccOmp5
    // - AccCpuTbbBlocks
    // - AccCpuSerial
    // using Acc = alpaka::AccCpuSerial<Dim, Idx>;
    using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;
    // Defines the synchronization behavior of a queue
    //
    // choose between Blocking and NonBlocking
    using AccQueueProperty = alpaka::Blocking;
    using DevQueue = alpaka::Queue<Acc, AccQueueProperty>;

    // Define the device accelerator
    //
    // It is possible to choose from a set of accelerators:
    // - AccCpuThreads
    // - AccCpuFibers
    // - AccCpuOmp2Threads
    // - AccCpuOmp2Blocks
    // - AccOmp5
    // - AccCpuSerial
    using Host = alpaka::AccCpuSerial<Dim, Idx>;
    // Defines the synchronization behavior of a queue
    //
    // choose between Blocking and NonBlocking
    using HostQueueProperty = alpaka::Blocking;
    using HostQueue = alpaka::Queue<Host, HostQueueProperty>;

    // Select devices
    auto const devAcc = alpaka::getDevByIdx<Acc>(0u);
    auto const devHost = alpaka::getDevByIdx<Host>(0u);

    // Create queues
    DevQueue devQueue(devAcc);
    HostQueue hostQueue(devHost);

    // Define the work division for kernels to be run on devAcc and devHost
    using Vec = alpaka::Vec<Dim, Idx>;
    Vec const elementsPerThread(Vec::all(static_cast<Idx>(1)));
    Vec const threadsPerGrid(Vec::all(static_cast<Idx>(10)));
    using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;
    WorkDiv const devWorkDiv = alpaka::getValidWorkDiv<Acc>(
        devAcc,
        threadsPerGrid,
        elementsPerThread,
        false,
        alpaka::GridBlockExtentSubDivRestrictions::Unrestricted);
    WorkDiv const hostWorkDiv = alpaka::getValidWorkDiv<Host>(
        devHost,
        threadsPerGrid,
        elementsPerThread,
        false,
        alpaka::GridBlockExtentSubDivRestrictions::Unrestricted);

    // Create host and device buffers
    //
    // A buffer is an n-dimensional structure with a
    // particular data type and size which corresponds
    // to memory on the desired device. Buffers can be
    // allocated on the device or can be obtained from
    // already existing allocations e.g. std::array,
    // std::vector or a simple call to new.
    using Data = std::uint32_t;
    constexpr Idx nElementsPerDim = 2;

    const Vec extents(Vec::all(static_cast<Idx>(nElementsPerDim)));

    // Allocate host memory buffers
    //
    // The `alloc` method returns a reference counted buffer handle.
    // When the last such handle is destroyed, the memory is freed automatically.
    using BufHost = alpaka::Buf<Host, Data, Dim, Idx>;
    BufHost hostBuffer(alpaka::allocBuf<Data, Idx>(devHost, extents));
    // You can also use already allocated memory and wrap it within a view (irrespective of the device type).
    // The view does not own the underlying memory. So you have to make sure that
    // the view does not outlive its underlying memory.
    std::array<Data, nElementsPerDim * nElementsPerDim * nElementsPerDim> plainBuffer;
    auto hostViewPlainPtr = alpaka::createView(devHost, plainBuffer.data(), extents);

    // Allocate accelerator memory buffers
    //
    // The interface to allocate a buffer is the same on the host and on the device.
    using BufAcc = alpaka::Buf<Acc, Data, Dim, Idx>;
    BufAcc deviceBuffer1(alpaka::allocBuf<Data, Idx>(devAcc, extents));
    BufAcc deviceBuffer2(alpaka::allocBuf<Data, Idx>(devAcc, extents));


    // Init host buffer
    //
    // You can not access the inner elements of a buffer directly, but you can get the pointer to the memory via
    // getPtrNative() or a read/write accessor using access().
    auto hostBufferAccessor = alpaka::experimental::access(hostBuffer);

    // This pointer can be used to directly write
    // some values into the buffer memory.
    // Mind, that only a host can write on host memory.
    // The same holds true for device memory.
    for(size_t z = 0; z < extents[0]; z++)
        for(size_t y = 0; y < extents[1]; y++)
            for(size_t x = 0; x < extents[2]; x++)
                hostBufferAccessor(z, y, x) = static_cast<Data>(alpaka::mapIdx<1u>(Vec{z, y, x}, extents)[0]);

    // Memory views and buffers can also be initialized by executing a kernel.
    // To pass a buffer into a kernel, you can pass the
    // native pointer into the kernel invocation.

    FillBufferKernel fillBufferKernel;
    alpaka::exec<Host>(hostQueue, hostWorkDiv, fillBufferKernel, alpaka::experimental::writeAccess(hostViewPlainPtr));

    // Copy host to device Buffer
    //
    // A copy operation of one buffer into
    // another buffer is enqueued into a queue
    // like it is done for kernel execution.
    // As always within alpaka, you will get a compile
    // time error if the desired copy coperation
    // (e.g. between various accelerator devices) is
    // not currently supported.
    // In this example both host buffers are copied
    // into device buffers.
    alpaka::memcpy(devQueue, deviceBuffer1, hostViewPlainPtr, extents);
    alpaka::memcpy(devQueue, deviceBuffer2, hostBuffer, extents);

    // Test device Buffer
    //
    // This kernel tests if the copy operations
    // were successful. In the case something
    // went wrong an assert will fail.

    TestBufferKernel testBufferKernel;
    alpaka::exec<Acc>(devQueue, devWorkDiv, testBufferKernel, alpaka::experimental::readAccess(deviceBuffer1));
    alpaka::exec<Acc>(devQueue, devWorkDiv, testBufferKernel, alpaka::experimental::readAccess(deviceBuffer2));

    // Print device Buffer
    //
    // Because we really like to flood our
    // terminal with numbers, the following
    // kernel prints all numbers of the
    // device buffer to stdout on the terminal.
    // Since this possibly is a parallel operation,
    // the output can appear in any order or even
    // completely distorted.

    PrintBufferKernel printBufferKernel;
    alpaka::exec<Acc>(devQueue, devWorkDiv, printBufferKernel, alpaka::experimental::readAccess(deviceBuffer1));
    alpaka::wait(devQueue);
    std::cout << std::endl;

    alpaka::exec<Host>(hostQueue, hostWorkDiv, printBufferKernel, alpaka::experimental::readAccess(hostBuffer));
    alpaka::wait(hostQueue);
    std::cout << std::endl;

    alpaka::exec<Host>(hostQueue, hostWorkDiv, printBufferKernel, alpaka::experimental::readAccess(hostViewPlainPtr));
    alpaka::wait(hostQueue);
    std::cout << std::endl;

    return EXIT_SUCCESS;
#endif
}
