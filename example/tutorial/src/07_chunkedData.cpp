/* Copyright 2024 Andrea Bocci, René Widera, Tim Hanel
 * SPDX-License-Identifier: Apache-2.0
 */

#include "config.h"

#include <alpaka/alpaka.hpp>

#include <cstdio>
#include <random>

/** @file
 *
 * In the previous example we showed how to handle thread indices by hand to iterate over 1 and 3-dimensional data.
 * There are very seldom cases where you need this explicit control over threads and blocks. Very often handling thread
 * indices by hand will result in performance issues at least on CPU devices.
 *
 * This example will show how you can iterate with frames, which can be seen as data index chunks without explicit
 * calculate thread indices by hand. The code is easy to write and read and will mostly be faster on CPU and GPU
 * devices.
 */

struct VectorAddKernel1D
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        alpaka::concepts::CVector auto chunkSize,
        alpaka::concepts::IDataSource auto const& in1,
        alpaka::concepts::IDataSource auto const& in2,
        alpaka::concepts::IMdSpan auto out) const
    {
        /* This kernel is called with 1- dimensional frame extents, nevertheless we will linearize the indices
         * explicitly which allow calling this kernel with M-dimensional frames extents too.
         *
         * All thread blocks will be used to iterate over the frames. Each thread block will handle one or more frames.
         */
        for(auto chunkOffset : alpaka::onAcc::makeIdxMap(
                acc,
                alpaka::onAcc::worker::linearBlocksInGrid,
                alpaka::IdxRange{Vec1D{0}, in1.getExtents(), chunkSize}))
        {
            /* We will use a 1-dimensional shared memory array to load the data from in1 for the corresponding frame
             * into it and use the cached data to compute the final result. Take care, shared memory is never
             * initialized and will contain garbage after the creation.
             *
             * The extents of a shared memory array must be known at compile time, therefore it is required that the
             * frame extent on the host side is of the type CVec.
             */
            auto sharedIn1Data = alpaka::onAcc::declareSharedMdArray<float, alpaka::uniqueId()>(acc, chunkSize);

            // iterate over elements within the frame and use all threads from the thread block
            for(auto inChunkIdx : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::linearThreadsInBlock,
                    alpaka::IdxRange{chunkSize}))
            {
                auto const globalDataIdx = chunkOffset + inChunkIdx;
                sharedIn1Data[inChunkIdx] = in1[globalDataIdx];
            }

            /* The synchronization is required because we will use for the second loop over frame element indicis a
             * different traversing schema. Therefore, you should not assume any thread and data element relation.
             */
            alpaka::onAcc::syncBlockThreads(acc);

            for(auto inChunkIdx : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::linearThreadsInBlock,
                    alpaka::IdxRange{chunkSize},
                    alpaka::onAcc::traverse::tiled))
            {
                auto const globalDataIdx = chunkOffset + inChunkIdx;
                out[globalDataIdx] = sharedIn1Data[inChunkIdx] + in2[globalDataIdx];
            }

            /* Synchronization is required if the same thread block is calculating more than one frame.
             * If this synchronization is missing threads can begin already to fill the shared memory with the next
             * frame data which results into a data race.
             */
            alpaka::onAcc::syncBlockThreads(acc);
        }
    }
};

struct VectorAddKernel3D
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        alpaka::concepts::CVector auto chunkExtents,
        alpaka::concepts::IDataSource auto const& in1,
        alpaka::concepts::IDataSource auto const& in2,
        alpaka::concepts::IMdSpan auto out) const
    {
        /* We will use a 3-dimensional shared memory array to load the data from in1 for the corresponding frame into
         * it and use the cached data to compute the final result. Take care, shared memory is never initialized and
         * will contain garbage after the creation.
         *
         * The extents of a shared memory array must be known at compile time, therefore it is required that the frame
         * extent on the host side is of the type CVec.
         */
        auto sharedIn1Data = alpaka::onAcc::declareSharedMdArray<float, alpaka::uniqueId()>(acc, chunkExtents);

        /* This kernel is called with 1- dimensional frame extents, nevertheless we will linearize the indices
         * explicitly which allow calling this kernel with M-dimensional frames extents too.
         *
         * All thread blocks will be used to iterate over the frames. Each thread block will handle one or more frames.
         */
        for(auto chunkOffset : alpaka::onAcc::makeIdxMap(
                acc,
                alpaka::onAcc::worker::blocksInGrid,
                alpaka::IdxRange{Vec3D::fill(0), in1.getExtents(), chunkExtents}))
        {
            // iterate over elements within the frame and use all threads from the thread block
            for(auto inChunkIdx :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInBlock, alpaka::IdxRange{chunkExtents}))
            {
                auto const globalDataIdxMD = chunkOffset + inChunkIdx;
                sharedIn1Data[inChunkIdx] = in1[globalDataIdxMD];
            }

            /* The synchronization is required because we will use for the second loop over frame element indicis a
             * different traversing schema. Therefore, you should not assume any thread and data element relation.
             */
            alpaka::onAcc::syncBlockThreads(acc);

            for(auto inChunkIdx : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::threadsInBlock,
                    alpaka::IdxRange{chunkExtents},
                    alpaka::onAcc::traverse::tiled))
            {
                auto const globalDataIdxMD = chunkOffset + inChunkIdx;
                out[globalDataIdxMD] = sharedIn1Data[inChunkIdx] + in2[globalDataIdxMD];
            }

            /* Synchronization is required if the same thread block is calculating more than one frame.
             * If this synchronization is missing threads can begin already to fill the shared memory with the next
             * frame data which results into a data race.
             */
            alpaka::onAcc::syncBlockThreads(acc);
        }
    }
};

void testVectorAddKernel(alpaka::onHost::concepts::Device auto device, auto computeExec)
{
    // random number generator with a gaussian distribution
    std::random_device rd{};
    std::default_random_engine rand{rd()};
    std::normal_distribution<float> dist{0.f, 1.f};

    // tolerance
    constexpr float epsilon = 0.000001f;

    // buffer size
    constexpr uint32_t size = 1024 * 1024;

    // allocate input and output host buffers in pinned memory accessible by the Platform devices
    auto in1_h = alpaka::onHost::allocHost<float>(size);
    auto in2_h = alpaka::onHost::allocHostLike(in1_h);
    auto out_h = alpaka::onHost::allocHostLike(in1_h);

    // fill the input buffers with random data, and the output buffer with zeros
    for(uint32_t i = 0; i < size; ++i)
    {
        in1_h[i] = dist(rand);
        in2_h[i] = dist(rand);
        out_h[i] = 0.;
    }

    // run the test the given device
    alpaka::onHost::Queue queue = device.makeQueue();

    // allocate input and output buffers on the device
    auto in1_d = alpaka::onHost::allocLike(device, in1_h);
    auto in2_d = alpaka::onHost::allocLike(device, in2_h);
    auto out_d = alpaka::onHost::allocLike(device, out_h);

    // copy the input data to the device; the size is known from the buffer objects
    alpaka::onHost::memcpy(queue, in1_d, in1_h);
    alpaka::onHost::memcpy(queue, in2_d, in2_h);

    // fill the output buffer with zeros; the size is known from the buffer objects
    alpaka::onHost::memset(queue, out_d, 0x00);

    // launch the 1-dimensional kernel
    constexpr auto chunkSize = 32u;
    auto numFrames = size / chunkSize;
    // The kernel assumes that the problem size is a multiple of the frame size.
    verify((numFrames * chunkSize) == size);

    auto frameSpec = alpaka::onHost::FrameSpec{numFrames, chunkSize, computeExec};

    // fill the output buffer with zeros; the size is known from the buffer objects
    alpaka::onHost::memset(queue, out_d, 0x00);

    std::cout << "Testing VectorAddKernel with vector indices with a grid of " << frameSpec << "\n";
    queue.enqueue(frameSpec, VectorAddKernel1D{}, alpaka::CVec<uint32_t, chunkSize>{}, in1_d, in2_d, out_d);

    // copy the results from the device to the host
    alpaka::onHost::memcpy(queue, out_h, out_d);

    // wait for all the operations to complete
    alpaka::onHost::wait(queue);

    // check the results
    for(uint32_t i = 0; i < size; ++i)
    {
        float sum = in1_h[i] + in2_h[i];
        verify(out_h[i] < sum + epsilon);
        verify(out_h[i] > sum - epsilon);
    }
    std::cout << "success\n";
}

void testVectorAddKernel3D(alpaka::onHost::concepts::Device auto device, auto computeExec)
{
    // random number generator with a gaussian distribution
    std::random_device rd{};
    std::default_random_engine rand{rd()};
    std::normal_distribution<float> dist{0., 1.};

    // tolerance
    constexpr float epsilon = 0.000001;

    // 3-dimensional and linearised buffer size
    constexpr Vec3D ndsize = {64, 128, 16};
    constexpr uint32_t size = ndsize.product();

    // allocate input and output host buffers in pinned memory accessible by the Platform devices
    auto in1_h = alpaka::onHost::allocHost<float>(ndsize);
    auto in2_h = alpaka::onHost::allocHostLike(in1_h);
    auto out_h = alpaka::onHost::allocHostLike(in1_h);

    // fill the input buffers with random data, and the output buffer with zeros
    for(uint32_t i = 0; i < size; ++i)
    {
        auto lIdx = alpaka::mapToND(ndsize, i);
        in1_h[lIdx] = dist(rand);
        in2_h[lIdx] = dist(rand);
        out_h[lIdx] = 0.;
    }

    // run the test the given device
    alpaka::onHost::Queue queue = device.makeQueue();

    // allocate input and output buffers on the device
    auto in1_d = alpaka::onHost::allocLike(device, in1_h);
    auto in2_d = alpaka::onHost::allocLike(device, in2_h);
    auto out_d = alpaka::onHost::allocLike(device, out_h);

    // copy the input data to the device; the size is known from the buffer objects
    alpaka::onHost::memcpy(queue, in1_d, in1_h);
    alpaka::onHost::memcpy(queue, in2_d, in2_h);

    // fill the output buffer with zeros; the size is known from the buffer objects
    alpaka::onHost::memset(queue, out_d, 0x00);


    // launch the 3-dimensional kernel
    auto chunkExtents = alpaka::CVec<uint32_t, 4, 4, 2>{};
    auto numFrames = ndsize / chunkExtents;
    // The kernel assumes that the problem size is a multiple of the frame size.
    verify((numFrames * chunkExtents).product() == size);

    auto frameSpec = alpaka::onHost::FrameSpec{numFrames, chunkExtents, computeExec};

    std::cout << "Testing VectorAddKernel3D with vector indices with a grid of " << frameSpec << "\n";

    queue.enqueue(frameSpec, VectorAddKernel3D{}, chunkExtents, in1_d, in2_d, out_d);

    // copy the results from the device to the host
    alpaka::onHost::memcpy(queue, out_h, out_d);

    // wait for all the operations to complete
    alpaka::onHost::wait(queue);

    // check the results
    for(uint32_t i = 0; i < size; ++i)
    {
        auto lIdx = alpaka::mapToND(ndsize, i);
        float sum = in1_h[lIdx] + in2_h[lIdx];
        verify(out_h[lIdx] < sum + epsilon);
        verify(out_h[lIdx] > sum - epsilon);
    }
    std::cout << "success\n";
}

int example(auto const deviceSpec, auto const computeExec)
{
    auto devSelector = alpaka::onHost::makeDeviceSelector(deviceSpec);

    // require at least one device
    std::size_t n = devSelector.getDeviceCount();

    if(n == 0)
    {
        return EXIT_FAILURE;
    }

    // use the first device
    alpaka::onHost::Device device = devSelector.makeDevice(0);
    std::cout << "Device: " << alpaka::onHost::getName(device) << "\n\n";

    testVectorAddKernel(device, computeExec);
    testVectorAddKernel3D(device, computeExec);

    return EXIT_SUCCESS;
}

auto main() -> int
{
    using namespace alpaka;

    /* Execute the example once for each backend (device specification + executor)
     *
     * If you would like to execute it for a single accelerator only you can use the following code.
     *  @code{.cpp}
     *  auto deviceSpec = onHost::DeviceSpec{api::cuda, deviceKind::nvidiaGpu};
     *  auto executor = exec::gpuCuda;
     *  return example(deviceSpec, executor, numElements);
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
     * https://alpaka3.readthedocs.io/en/latest/basic/cheatsheet.html#available-apis
     * A list of executors can be found
     * https://alpaka3.readthedocs.io/en/latest/basic/cheatsheet.html#executors
     */
    return onHost::executeForEach(
        [=](alpaka::concepts::BackendSpec auto const& backend)
        {
            auto selector = onHost::makeDeviceSelector(alpaka::onHost::DeviceSpec{backend});
            if(!selector.isAvailable())
                return EXIT_SUCCESS;
            return example(alpaka::onHost::DeviceSpec{backend}, alpaka::getExecutor(backend));
        },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
