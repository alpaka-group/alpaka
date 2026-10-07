/* Copyright 2026  René Widera, Tim Hanel
 * SPDX-License-Identifier: ISC
 */

/** @file This example demonstrates how to use alpaka's function interface.
 *
 * The function interface is a mechanism to register and dispatch a function interface from alpaka to third party
 * libraries for specific APIs and device kinds.
 *
 * The alpaka function symbols provides you with a clean unified way to dispatch a user chosen function interface
 * signature to functions of different vendors. There is no need to deal with preprocessor macros on the caller side
 * within user functions, which keeps the code readable. You could create quite similar results with C++ functor trait
 * specializations but with the drawback that you must deal with type signature on the caller side instead of having a
 * well-defined way how the function call is dispatched to the best fitting overload.
 *
 * The example shows how to register and dispatch a vendor function for the api Host and device kind Cpu, which is
 * implemented with std::transform. If available, a vendor function overload for the api Cuda and device kind NvidiaGpu
 * is registered as well, which is implemented with thrust::transform. If no vendor function overload is registered for
 * the given queue or device specification, a generic fallback implementation using alpaka kernels is called.
 */

#include "alpakaFn.hpp"
#include "cudaFn.hpp"
#include "fn.hpp"

#include <alpaka/alpaka.hpp>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <typeinfo>

using namespace alpaka;

auto validateResult(auto const& input0, auto const& input1, auto const& output) -> int
{
    using Data = uint32_t;
    static constexpr int MAX_PRINT_FALSE_RESULTS = 20;
    int falseResults = 0;
    for(std::size_t i(0u); i < output.getExtents(); ++i)
    {
        Data const& val(output[i]);
        Data const correctResult(input0[i] + input1[i]);
        if(val != correctResult)
        {
            if(falseResults < MAX_PRINT_FALSE_RESULTS)
                std::cerr << "C[" << i << "] == " << val << " != " << correctResult << std::endl;
            ++falseResults;
        }
    }

    if(falseResults == 0)
        return EXIT_SUCCESS;

    std::cout << "Found " << falseResults << " false results, printed no more than " << MAX_PRINT_FALSE_RESULTS << "\n"
              << "Execution results incorrect!" << std::endl;
    std::cout << std::endl;
    return EXIT_FAILURE;
}

// In standard projects, you typically do not execute the code with any available accelerator.
// Instead, a single accelerator is selected once from the active accelerators and the kernels are executed with the
// selected accelerator only. If you use the example as the starting point for your project, you can rename the
// example() function to main() and move the accelerator tag to the function body.
auto example(auto const deviceSpec, auto const exec, size_t numElements) -> int
{
    using IdxVec = Vec<std::size_t, 1u>;

    // Define problem size
    IdxVec const extent(numElements);

    // Define the buffer element type
    using Data = uint32_t;

    std::cout << "Number of elements: " << numElements << std::endl;
    std::cout << "Element type: " << onHost::demangledName<Data>() << std::endl;

    std::cout << "Using alpaka accelerator: " << onHost::demangledName(exec) << " for "
              << deviceSpec.getApi().getName() << " " << deviceSpec.getDeviceKind().getName() << std::endl;

    // Select a device
    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device devAcc = devSelector.makeDevice(0);

    // Create a queue on the device
    onHost::Queue queue = devAcc.makeQueue();

    // Allocate 3 host memory buffers
    auto iotaBuffer_h = onHost::allocHost<Data>(extent);
    auto output_h = onHost::allocHostLike(iotaBuffer_h);

    for(auto i(0u); i < iotaBuffer_h.getExtents().x(); ++i)
        iotaBuffer_h[i] = i;

    // Allocate 3 buffers on the accelerator
    auto input0_d = onHost::allocLike(devAcc, iotaBuffer_h);
    auto input1_d = onHost::allocLike(devAcc, iotaBuffer_h);
    auto output_d = onHost::allocLike(devAcc, iotaBuffer_h);

    // copy input data to device and clear device output buffer
    onHost::memcpy(queue, input0_d, iotaBuffer_h);
    onHost::memcpy(queue, input1_d, iotaBuffer_h);
    onHost::fill(queue, output_d, Data{0});

    vendorExample::Transform::call(queue, output_d, std::plus{}, input0_d, input1_d);

    // copy results back to host
    onHost::memcpy(queue, output_h, output_d);
    onHost::wait(queue);

    if(int const result = validateResult(iotaBuffer_h, iotaBuffer_h, output_h); result != EXIT_SUCCESS)
        return result;

    std::cout << "Execution results correct!" << std::endl;
    std::cout << std::endl;
    return EXIT_SUCCESS;
}

void help(char* argv[])
{
    std::cerr << argv[0] << " [-n  numElements] [-h]" << std::endl;
}

auto main(int argc, char* argv[]) -> int
{
    size_t numElements = 128;

    int opt;
    while((opt = getopt(argc, argv, "hn:")) != -1)
    {
        switch(opt)
        {
        case 'n':
            try
            {
                numElements = std::stoul(optarg, nullptr, 0);
            }
            catch(std::invalid_argument const& e)
            {
                std::cerr << "Error: invalid argument '" << optarg << "'.\n";
                return EXIT_FAILURE;
            }
            catch(std::out_of_range const& e)
            {
                std::cerr << "Error: value '" << optarg << "' out of range for size_t.\n";
                return EXIT_FAILURE;
            }
            break;
        case 'h':
            help(argv);
            exit(EXIT_SUCCESS);
        default:
            help(argv);
            exit(EXIT_FAILURE);
        }
    }

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
            return example(alpaka::onHost::DeviceSpec{backend}, alpaka::getExecutor(backend), numElements);
        },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
