/* Copyright 2025 Anton Reinhard, Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <iostream>

using IdxType = uint32_t;

using namespace alpaka;

/** @brief An example boundary kernel that copies only the values in the given boundary direction
 */
struct BoundaryExampleKernel
{
    ALPAKA_FN_ACC auto operator()(
        auto const& acc,
        concepts::IMdSpan auto const& view,
        concepts::IMdSpan auto viewTarget,
        concepts::BoundaryDirection auto const& bd) const
    {
        for(auto idx : onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, IdxRange{view.getExtents()}, bd))
        {
            viewTarget[idx] = view[idx];
        }
    }
};

auto printBoundaryContainer(auto view, auto bd_container)
{
    std::cout << "Number of unique boundaries: " << bd_container.length() << std::endl;
    for(auto bd : bd_container)
    {
        // create the sub view for this boundary direction
        auto subView = view.getSubView(bd);
        std::cout << bd << ":";
        std::cout << " extent: " << subView.getExtents() << " elements: ";

        // show sub view's values
        for(auto const& el : subView)
            std::cout << el << " ";
        std::cout << std::endl;
    }

    std::cout << "End of Boundaries\n\n" << std::endl;
}

int example(auto const devSpec, auto const computeExec)
{
    // initialise the accelerator platform
    auto devSelector = onHost::makeDeviceSelector(devSpec);

    // require at least one device
    [[maybe_unused]] std::size_t n = devSelector.getDeviceCount();
    assert(n > 0);
    auto device = devSelector.makeDevice(0);

    // set example buffer size
    constexpr IdxType Dimensions = 4;
    constexpr auto size = alpaka::fillCVec<IdxType, Dimensions, 4>();

    // allocate a buffer of floats in host memory
    auto buffer = onHost::alloc<float>(device, size);
    auto const view = buffer.getView();

    auto bufferTarget = onHost::allocLike(device, buffer);
    auto viewTarget = buffer.getView();

    auto blockingQueue = device.makeQueue(queueKind::blocking);
    onHost::iota(blockingQueue, 1.f, buffer);

    std::cout << "Running on device: " << device.getName() << std::endl;
    std::cout << Dimensions << "D volume" << std::endl;
    std::cout << "Volume Extents: " << view.getExtents() << std::endl;

    // show all values once
    if constexpr(std::is_same_v<ALPAKA_TYPEOF(device), onHost::Device<api::Host, deviceKind::Cpu>>)
    {
        for(auto& el : view)
            std::cout << el << " ";
        std::cout << std::endl;
    }

    auto chunkSize = alpaka::fillCVec<IdxType, Dimensions, 2>();
    auto frameSpec = onHost::FrameSpec{divCeil(size, chunkSize), chunkSize, computeExec};

    auto exampleKernel = BoundaryExampleKernel{};


    {
        std::cout << "Halo size 1 (default) for " << Dimensions << "D" << std::endl;

        constexpr auto bdContainer = makeBoundaryDirIterator<Dimensions>();

        if constexpr(std::is_same_v<ALPAKA_TYPEOF(device), onHost::Device<api::Host, deviceKind::Cpu>>)
        {
            printBoundaryContainer(view, bdContainer);
        }

        auto const bd = makeCoreBoundaryDirection<Dimensions>();
        blockingQueue.enqueue(frameSpec, KernelBundle{exampleKernel, view, viewTarget, bd});
    }

    {
        std::cout << "Halo size 1 (default), constructed directly from the view" << std::endl;

        constexpr auto bdContainer = makeBoundaryDirIterator(view);

        if constexpr(std::is_same_v<ALPAKA_TYPEOF(device), onHost::Device<api::Host, deviceKind::Cpu>>)
        {
            printBoundaryContainer(view, bdContainer);
        }
    }

    {
        std::cout << "Halo size 2" << std::endl;

        auto const halo = alpaka::fillCVec<IdxType, Dimensions, 2>();

        constexpr auto bdContainerHalo = makeBoundaryDirIterator(halo);
        if constexpr(std::is_same_v<ALPAKA_TYPEOF(device), onHost::Device<api::Host, deviceKind::Cpu>>)
        {
            printBoundaryContainer(view, bdContainerHalo);
        }

        auto const bd = makeCoreBoundaryDirection<Dimensions>(halo);
        blockingQueue.enqueue(frameSpec, KernelBundle{exampleKernel, view, viewTarget, bd});
    }

    {
        std::cout << "Lower halo sizes = 1, upper halo sizes = 2" << std::endl;

        auto const lowerHalo = alpaka::fillCVec<IdxType, Dimensions, 1>();
        auto const upperHalo = alpaka::fillCVec<IdxType, Dimensions, 2>();

        constexpr auto bdContainerHaloHeterogeneous = makeBoundaryDirIterator(lowerHalo, upperHalo);

        if constexpr(std::is_same_v<ALPAKA_TYPEOF(device), onHost::Device<api::Host, deviceKind::Cpu>>)
        {
            printBoundaryContainer(view, bdContainerHaloHeterogeneous);
        }

        auto const bd = makeCoreBoundaryDirection<Dimensions>(lowerHalo, upperHalo);
        blockingQueue.enqueue(frameSpec, KernelBundle{exampleKernel, view, viewTarget, bd});
    }

    return EXIT_SUCCESS;
}

auto main() -> int
{
    /* Execute the example once for each backend (device specification + executor)
     *
     * If you would like to execute it for a single accelerator only you can use the following code.
     *  @code{.cpp}
     *  auto deviceSpec = onHost::DeviceSpec{api::cuda, deviceKind::nvidiaGpu};
     *  auto executor = exec::gpuCuda;
     *  return example(deviceSpec, executor);
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
