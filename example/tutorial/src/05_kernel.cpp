/* Copyright 2024 Andrea Bocci, René Widera, Tim Hanel
 * SPDX-License-Identifier: Apache-2.0
 */

#include "config.h"

#include <alpaka/alpaka.hpp>

#include <cstdio>
#include <random>

struct VectorAddKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        alpaka::concepts::IDataSource auto const& in1,
        alpaka::concepts::IDataSource auto const& in2,
        alpaka::concepts::IMdSpan auto out,
        uint32_t size) const
    {
        for(auto [index] :
            alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{size}))
        {
            out[index] = in1[index] + in2[index];
        }
    }
};

struct VectorAddKernel1D
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        alpaka::concepts::IDataSource auto const& in1,
        alpaka::concepts::IDataSource auto const& in2,
        alpaka::concepts::IMdSpan auto out,
        Vec1D size) const
    {
        for(auto ndindex :
            alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{size}))
        {
            out[ndindex] = in1[ndindex] + in2[ndindex];
        }
    }
};

struct VectorAddKernel3D
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        alpaka::concepts::IDataSource auto const& in1,
        alpaka::concepts::IDataSource auto const& in2,
        alpaka::concepts::IMdSpan auto out,
        Vec3D size) const
    {
        for(auto ndindex :
            alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{size}))
        {
            out[ndindex] = in1[ndindex] + in2[ndindex];
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

    // launch the 1-dimensional kernel with scalar size
    auto frameSpec = alpaka::onHost::FrameSpec{32u, 32u, computeExec};

    std::cout << "Testing VectorAddKernel with scalar indices with a grid of " << frameSpec << "\n";
    queue.enqueue(frameSpec, VectorAddKernel{}, in1_d, in2_d, out_d, size);

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

    // fill the output buffer with zeros; the size is known from the buffer objects
    alpaka::onHost::memset(queue, out_d, 0x00);


    std::cout << "Testing VectorAddKernel with vector indices with a grid of " << frameSpec << "\n";
    queue.enqueue(frameSpec, VectorAddKernel1D{}, in1_d, in2_d, out_d, Vec1D{size});

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
    constexpr Vec3D ndsize = {50, 125, 16};
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
    auto frameSpec = alpaka::onHost::FrameSpec{Vec3D{5, 5, 1}, Vec3D{4, 4, 4}, computeExec};
    std::cout << "Testing VectorAddKernel3D with vector indices with a grid of " << frameSpec << "\n";

    queue.enqueue(frameSpec, VectorAddKernel3D{}, in1_d, in2_d, out_d, ndsize);

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

    /* This example requires additionally to the device specification an executor to describe how the parallelism
     * on the compute device is organised.
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
     * https://alpaka.readthedocs.io/en/latest/basic/cheatsheet.html#available-apis
     * A list of executors can be found
     * https://alpaka.readthedocs.io/en/latest/basic/cheatsheet.html#executors
     */
    example(onHost::DeviceSpec{api::host, deviceKind::cpu}, exec::cpuSerial);

    // Execute the example once for each enabled API, device kind, and executor.
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
