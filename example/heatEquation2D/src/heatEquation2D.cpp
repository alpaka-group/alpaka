/* Copyright 2020 Benjamin Worpitz, Matthias Werner, Jakob Krude, Sergei Bastrakov, Bernhard Manfred Gruber,
 * Tapish Narwal, Anton Reinhard, Tim Hanel
 * SPDX-License-Identifier: ISC
 */

#include "BoundaryKernel.hpp"
#include "StencilKernel.hpp"
#include "alpaka/onHost/FrameSpec.hpp"
#include "analyticalSolution.hpp"

#ifdef PNGWRITER_ENABLED
#    include "writeImage.hpp"
#endif

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <utility>

namespace alpaka::example::heatEquation
{
    using IdxType = uint32_t;
    using Data = double;

    void printExampleHeader(
        IdxType const sideLength,
        IdxType const numTimeSteps,
        bool const autoCheck,
        double const tMax)
    {
        std::cout << "================================" << std::endl;
        std::cout << "Example Heat Equation Stencil" << std::endl;
        std::cout << "    Number of elements [#]: " << sideLength << " x " << sideLength << " = "
                  << sideLength * sideLength << std::endl;
        std::cout << "    Time steps [#]: " << numTimeSteps << std::endl;
        std::cout << "    tMax: " << tMax << std::endl;
        std::cout << "    Element type: " << onHost::demangledName<Data>() << std::endl;
        std::cout << "    Buffer size [kbyte]: " << sideLength * sideLength * sizeof(Data) / 1.e3 << std::endl;
        std::cout << "    Result checking is " << (autoCheck ? "enabled" : "disabled") << std::endl;
        std::cout << "================================" << std::endl;
        std::cout << std::endl;
    }

    //! Each kernel computes the next step for one point.
    //! Therefore, the number of threads should be equal to numNodesX.
    //! Every time step the kernel will be executed numNodesX-times
    //! After every step the curr-buffer will be set to the calculated values
    //! from the next-buffer.
    //!
    //! In standard projects, you typically do not execute the code with any available accelerator.
    //! Instead, a single accelerator is selected once from the active accelerators and the kernels are executed with
    //! the selected accelerator only. If you use the example as the starting point for your project, you can rename
    //! the example() function to main() and move the accelerator tag to the function body.
    int example(
        auto const deviceSpec,
        auto const computeExec,
        uint32_t const sideLength,
        uint32_t const numTimeSteps,
        double const tMax,
        bool const enableCheck)
    {
        using namespace alpaka;
        using namespace alpaka::onHost;

        constexpr uint32_t Dimensions = 2u;
        using IdxTypeVec = Vec<IdxType, Dimensions>;

        std::cout << deviceSpec.getApi().getName() << std::endl;

        auto devSelector = makeDeviceSelector(deviceSpec);
        Device devAcc = devSelector.makeDevice(0);

#if ALPAKA_LANG_ONEAPI
        // support for double precision is not guaranteed for sycl devices such as Intel GPUs
        if constexpr(deviceSpec.getApi() == api::oneApi && std::is_same_v<Data, double>)
        {
            if(devAcc.getNativeHandle().first.template get_info<sycl::info::device::double_fp_config>().size() == 0)
            {
                std::cout << getName(devAcc) << " does not support double precision"
                          << "\n";
                std::cout << "Skip benchmark.\n";
                std::cout << "For Intel Arc GPUs, use the environment variables `IGC_EnableDPEmulation=1 "
                             "OverrideDefaultFP64Settings=1` to emulate double precision support.\n";
                // return 0 otherwise ctest fails
                return EXIT_SUCCESS;
            }
        }
#endif

        // simulation defines
        // {Y, X}
        IdxTypeVec const numNodes{sideLength, sideLength};
        IdxTypeVec const haloSize{1, 1};
        IdxTypeVec const extent = numNodes + haloSize + haloSize;

        // x, y in [0, 1], t in [0, tMax]
        Data const dx = 1.0 / static_cast<Data>(extent[1] - 1);
        Data const dy = 1.0 / static_cast<Data>(extent[0] - 1);
        Data const dt = tMax / static_cast<Data>(numTimeSteps);

        // Check the stability condition
        Data const r = 2 * dt / ((dx * dx * dy * dy) / (dx * dx + dy * dy));
        if(r > 1.)
        {
            std::cerr << "Stability condition check failed: dt/min(dx^2,dy^2) = " << r
                      << ", it is required to be <= 0.5\n";
            return EXIT_FAILURE;
        }

        // Initialize host-buffer
        // This buffer will hold the current values (used for the next step)
        auto uBufHost = allocHost<Data>(extent);

        // Accelerator buffer
        auto uCurrBufAcc = allocLike(devAcc, uBufHost);
        auto uNextBufAcc = allocLike(devAcc, uBufHost);

        // Set buffer to initial conditions
        initalizeBuffer(uBufHost.getMdSpan(), dx, dy);

        // Select queue
        Queue dumpQueue = devAcc.makeQueue();
        Queue computeQueue = devAcc.makeQueue();

        // Copy host -> device
        memcpy(computeQueue, uCurrBufAcc, uBufHost);
        wait(computeQueue);

        // Appropriate chunk size to split your problem for your Acc
        constexpr IdxType xSize = 16u;
        constexpr IdxType ySize = 16u;
        constexpr IdxType halo = 2u;
        constexpr auto chunkSize = CVec<IdxType, ySize, xSize>{};
        auto const numNodesWithHalo = numNodes + halo;

        IdxTypeVec const numChunks{
            divCeil(numNodes[0], chunkSize[0]),
            divCeil(numNodes[1], chunkSize[1]),
        };

        assert(
            numNodes[0] % chunkSize[0] == 0 && numNodes[1] % chunkSize[1] == 0
            && "Domain must be divisible by chunk size");

        auto sharedMemExtents = CVec<uint32_t, ySize + halo, xSize + halo>{};

        StencilKernel stencilKernel;
        BoundaryKernel boundaryKernel;

        auto const dataBlockingStencil = FrameSpec{numChunks, chunkSize, computeExec};

        auto const longestSide = std::max(numNodesWithHalo.y(), numNodesWithHalo.x());
        auto const dataBlockingBorder
            = FrameSpec{Vec{longestSide / chunkSize.x()}, Vec{std::max(chunkSize.y(), chunkSize.x())}, computeExec};

        auto const startTime = std::chrono::high_resolution_clock::now();

        // Simulate
        for(uint32_t step = 1; step <= numTimeSteps; ++step)
        {
            // Compute next values
            computeQueue.enqueue(
                dataBlockingStencil,
                KernelBundle{
                    stencilKernel,
                    uCurrBufAcc,
                    uNextBufAcc,
                    chunkSize,
                    sharedMemExtents,
                    numNodes,
                    dx,
                    dy,
                    dt});

            computeQueue.enqueue(
                dataBlockingBorder,
                KernelBundle{boundaryKernel, uNextBufAcc.getMdSpan(), numNodesWithHalo, step, dx, dy, dt});

#ifdef PNGWRITER_ENABLED
            if((step - 1) % 100 == 0)
            {
                wait(computeQueue);
                memcpy(dumpQueue, uBufHost, uCurrBufAcc);
                wait(dumpQueue);
                writeImage(step - 1, uBufHost.getMdSpan());
            }
#endif

            // So we just swap next and curr (shallow copy)
            std::swap(uNextBufAcc, uCurrBufAcc);
        }

        wait(computeQueue);
        auto const endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsedTime = endTime - startTime;

        std::cout << "Simulation took " << elapsedTime.count() << " seconds." << std::endl;
        std::cout << "Time per time step: " << elapsedTime.count() / numTimeSteps * 1000 << " ms." << std::endl;


        // Copy device -> host
        memcpy(dumpQueue, uBufHost, uCurrBufAcc);
        wait(dumpQueue);

        // Validate
        if(!enableCheck)
            return EXIT_SUCCESS;

        auto const [resultIsCorrect, maxError] = validateSolution(uBufHost.getMdSpan(), extent, dx, dy, tMax);

        if(resultIsCorrect)
        {
            std::cout << "Execution results correct!" << std::endl;
            return EXIT_SUCCESS;
        }
        else
        {
            std::cout << "Execution results incorrect: Max error = " << maxError
                      << " (the grid resolution may be too low)" << std::endl;
            return EXIT_FAILURE;
        }
    }

    void help(char* argv[])
    {
        std::cerr << argv[0] << " [OPTIONS]" << std::endl;
        std::cerr << "  -n sideLength: Number of elements per side of the 2D plane. Default: 2^6 = 64" << std::endl;
        std::cerr << "  -t numTimeSteps: Number of time steps that the simulation is run for. Default: 4000"
                  << std::endl;
        std::cerr << "  -d tMax: Delta t for the timesteps, has to be smaller for larger sideLengths to pass the "
                     "stability condition. Default: 0.1"
                  << std::endl;
        std::cerr << "  -c: disable checking for correct results" << std::endl;
        std::cerr << "  -h: Print this help message" << std::endl;
        std::cerr << std::endl;
    }

} // namespace alpaka::example::heatEquation

auto main(int argc, char* argv[]) -> int
{
    using namespace alpaka;
    using namespace alpaka::example::heatEquation;

    // Default value if no command line argument used
    IdxType sideLength = 1 << 6;
    IdxType numTimeSteps = 4000;

    int opt;
    bool enableCheck = true;
    double tMax = 0.1;

    while((opt = getopt(argc, argv, "hn:t:d:c")) != -1)
    {
        switch(opt)
        {
        case 'n':
            try
            {
                sideLength = static_cast<IdxType>(std::stoull(optarg, nullptr, 0));
            }
            catch(std::invalid_argument const& e)
            {
                std::cerr << "Error: invalid argument '" << optarg << "'.\n";
                return EXIT_FAILURE;
            }
            catch(std::out_of_range const& e)
            {
                std::cerr << "Error: value '" << optarg << "' out of range for unsigned long long.\n";
                return EXIT_FAILURE;
            }
            break;
        case 't':
            try
            {
                numTimeSteps = static_cast<IdxType>(std::stoull(optarg, nullptr, 0));
            }
            catch(std::invalid_argument const& e)
            {
                std::cerr << "Error: invalid argument '" << optarg << "'.\n";
                return EXIT_FAILURE;
            }
            catch(std::out_of_range const& e)
            {
                std::cerr << "Error: value '" << optarg << "' out of range for unsigned long long.\n";
                return EXIT_FAILURE;
            }
            break;
        case 'd':
            try
            {
                tMax = std::stod(optarg, nullptr);
            }
            catch(std::invalid_argument const& e)
            {
                std::cerr << "Error: invalid argument '" << optarg << "'.\n";
                return EXIT_FAILURE;
            }
            catch(std::out_of_range const& e)
            {
                std::cerr << "Error: value '" << optarg << "' out of range for unsigned long long.\n";
                return EXIT_FAILURE;
            }
            break;
        case 'h':
            help(argv);
            exit(EXIT_SUCCESS);
        case 'c':
            enableCheck = false;
            break;
        default:
            help(argv);
            exit(EXIT_FAILURE);
        }
    }

    printExampleHeader(sideLength, numTimeSteps, enableCheck, tMax);

    /* Execute the example once for each backend (device specification + executor)
     *
     * If you would like to execute it for a single accelerator only you can use the following code.
     *  @code{.cpp}
     *  auto deviceSpec = onHost::DeviceSpec{api::cuda, deviceKind::nvidiaGpu};
     *  auto executor = exec::gpuCuda;
     *  return alpaka::example::heatEquation::example(deviceSpec, executor, ...);
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
            return alpaka::example::heatEquation::example(
                alpaka::onHost::DeviceSpec{backend},
                alpaka::getExecutor(backend),
                sideLength,
                numTimeSteps,
                tMax,
                enableCheck);
        },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
