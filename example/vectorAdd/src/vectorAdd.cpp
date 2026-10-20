/* Copyright 2024  Jan Stephan, Luca Ferragina, Aurora Perego, Andrea Bocci, René Widera, Mehmet Yusufoglu, Tim Hanel.
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <typeinfo>

using namespace alpaka;

auto validateResult(auto const& bufHostC, auto const& bufHostA, auto const& bufHostB, std::size_t extent) -> int
{
    using Data = uint32_t;
    static constexpr int MAX_PRINT_FALSE_RESULTS = 20;
    int falseResults = 0;
    for(std::size_t i(0u); i < extent; ++i)
    {
        Data const& val(bufHostC[i]);
        Data const correctResult(bufHostA[i] + bufHostB[i]);
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

//! A vector addition kernel.
class VectorAddKernel
{
public:
    //! The kernel entry point.
    //!
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \tparam TElem The matrix element type.
    //! \param acc The accelerator to be executed on.
    //! \param A The first source vector.
    //! \param B The second source vector.
    //! \param C The destination vector.
    //! \param numElements The number of elements.
    ALPAKA_FN_ACC auto operator()(
        auto const& acc,
        alpaka::concepts::IMdSpan auto const A,
        alpaka::concepts::IMdSpan auto const B,
        alpaka::concepts::IMdSpan auto C,
        auto const& numElements) const -> void
    {
        using namespace alpaka;
        static_assert(ALPAKA_TYPEOF(numElements)::dim() == 1, "The VectorAddKernel expects 1-dimensional indices!");

        auto simdGrid = onAcc::SimdAlgo{onAcc::worker::threadsInGrid};
        simdGrid.concurrent(
            acc,
            numElements,
            [&](auto const&, auto&& simdA, auto&& simdB, auto&& simdC) constexpr
            { simdC = simdA.load() + simdB.load(); },
            A,
            B,
            C);
    }
};

// In standard projects, you typically do not execute the code with any available accelerator.
// Instead, a single accelerator is selected once from the active accelerators and the kernels are executed with the
// selected accelerator only. If you use the example as the starting point for your project, you can rename the
// example() function to main() and move the accelerator tag to the function body.
auto example(auto const deviceSpec, auto const exec, size_t numElements, size_t numberOfRuns) -> int
{
    using IdxVec = Vec<std::size_t, 1u>;

    // Define problem size
    IdxVec const extent(numElements);

    // Define the buffer element type
    using Data = uint32_t;

    std::cout << "Number of elements: " << numElements << std::endl;
    std::cout << "Element type: " << onHost::demangledName<Data>() << std::endl;
    std::cout << "Number of runs: " << numberOfRuns << std::endl;

    std::cout << "Using alpaka accelerator: " << onHost::demangledName(exec) << " for "
              << deviceSpec.getApi().getName() << " " << deviceSpec.getDeviceKind().getName() << std::endl;

    // Select a device
    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device devAcc = devSelector.makeDevice(0);

    // Create a queue on the device
    onHost::Queue queue = devAcc.makeQueue();

    // Allocate 3 host memory buffers
    auto bufHostA = onHost::allocHost<Data>(extent);
    auto bufHostB = onHost::allocHostLike(bufHostA);
    auto bufHostC = onHost::allocHostLike(bufHostA);

    // C++14 random generator for uniformly distributed numbers in {1,..,42}
    std::random_device rd{};
    std::default_random_engine eng{rd()};
    std::uniform_int_distribution<Data> dist(1, 42);

    for(auto i(0u); i < extent.x(); ++i)
    {
        bufHostA[i] = dist(eng);
        bufHostB[i] = dist(eng);
        bufHostC[i] = 0;
    }

    // Allocate 3 buffers on the accelerator
    auto bufAccA = onHost::allocLike(devAcc, bufHostA);
    auto bufAccB = onHost::allocLike(devAcc, bufHostB);
    auto bufAccC = onHost::allocLike(devAcc, bufHostC);

    Vec<size_t, 1u> chunkSize = 256u;
    // how many elements one worker should compute to ensure vectorization or instruction parallelism
    uint32_t elementsPerWorker = getNumElemPerThread<Data>(queue);
    auto dataBlocking = onHost::FrameSpec{divCeil(extent, chunkSize * elementsPerWorker), chunkSize, exec};

    // Instantiate the kernel function object
    VectorAddKernel kernel;
    auto const taskKernel = KernelBundle{kernel, bufAccA, bufAccB, bufAccC, extent};

    // Copy Host -> Acc
    onHost::memcpy(queue, bufAccA, bufHostA);
    onHost::memcpy(queue, bufAccB, bufHostB);

    double totalKernelRuntime = 0.0;
    double totalCopyRuntime = 0.0;
    std::size_t completedRuns = 0u;

    for(; completedRuns < numberOfRuns; ++completedRuns)
    {
        // set the device memory to all zeros (byte-wise, not element-wise)
        onHost::memset(queue, bufAccC, uint8_t{0});

        // Kernel execution timing
        onHost::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();
        // Enqueue the kernel execution task
        queue.enqueue(dataBlocking, taskKernel);
        // wait in case we are using an asynchronous queue to time actual kernel runtime
        onHost::wait(queue);
        auto const endT = std::chrono::high_resolution_clock::now();
        double kernelRuntime = std::chrono::duration<double>(endT - beginT).count();
        totalKernelRuntime += kernelRuntime;

        // Copy back the result
        auto beginCopyT = std::chrono::high_resolution_clock::now();
        onHost::memcpy(queue, bufHostC, bufAccC);
        onHost::wait(queue);
        auto const endCopyT = std::chrono::high_resolution_clock::now();
        double copyRuntime = std::chrono::duration<double>(endCopyT - beginCopyT).count();
        totalCopyRuntime += copyRuntime;

        if(completedRuns == 0)
        {
            if(int const result = validateResult(bufHostC, bufHostA, bufHostB, extent.x()); result != EXIT_SUCCESS)
                return result;
        }
    }

    if(completedRuns > 0u)
    {
        double avgKernelRuntime = totalKernelRuntime / static_cast<double>(completedRuns);
        double avgCopyRuntime = totalCopyRuntime / static_cast<double>(completedRuns);
        std::cout << "Average time for kernel execution: " << avgKernelRuntime << "s" << std::endl;
        std::cout << "Average time for HtoD copy: " << avgCopyRuntime << "s" << std::endl;
    }

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
    size_t numElements = 123456;
    size_t numberOfRuns = 1;

    int opt;
    while((opt = getopt(argc, argv, "hn:r:")) != -1)
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
        case 'r':
            try
            {
                numberOfRuns = std::stoul(optarg, nullptr, 0);
            }
            catch(std::invalid_argument const& e)
            {
                std::cerr << "Error: invalid number of runs '" << optarg << "'.\n";
                return EXIT_FAILURE;
            }
            catch(std::out_of_range const& e)
            {
                std::cerr << "Error: number of runs '" << optarg << "' out of range for size_t.\n";
                return EXIT_FAILURE;
            }
            if(numberOfRuns == 0)
            {
                std::cerr << "Error: number of runs must be greater than zero.\n";
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
            return example(
                alpaka::onHost::DeviceSpec{backend},
                alpaka::getExecutor(backend),
                numElements,
                numberOfRuns);
        },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
