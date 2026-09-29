/* Copyright 2024-2025 Tapish Narwal, Mehmet Yusufoglu, Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */

#include "randomInitNormal.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/executeForEach.hpp>

#include <cmath>
#include <iostream>
#include <vector>
constexpr uint32_t numberOfBins = 1024;
constexpr uint32_t numberOfElements = 8 * 1024 * 1024;
constexpr uint32_t blockSize = 1024;

// Function to validate the histogram
bool validateHistogram(auto const& histogram, int32_t const expectedValue)
{
    if(histogram.empty())
    {
        std::cerr << "Error: Histogram is empty.\n";
        return false;
    }

    // Define acceptable tolerance (allowed max-min difference in histogram array)
    float const tolerance = 0.1f;

    // Find min and max values in the histogram
    auto minmax = std::minmax_element(histogram.begin(), histogram.end());
    uint32_t minValue = *minmax.first;
    uint32_t maxValue = *minmax.second;

    // Calculate the allowed difference
    uint32_t allowedDifference = static_cast<uint32_t>(expectedValue * tolerance);

    // Check if the difference between max and min is within the tolerance
    bool isValid = (maxValue - minValue) <= allowedDifference;

    std::cerr << "Max value: " << maxValue << ", Min value: " << minValue << ", Difference: " << maxValue - minValue
              << ", Allowed difference: " << allowedDifference << "\n";
    if(!isValid)
    {
        std::cerr << "Histogram validation failed: Difference between max and min values is too large.\n";
    }

    return isValid;
}

//! Kernel to generate random numbers and fill the histogram
//! This kernel uses a uniform random integer generator and uses a bit manipulator-scaler to generate uniform float
//! random numbers in the range [0, 1). The generated random numbers are then scaled to fit into the histogram bins.
//! The histogram is filled by atomically incrementing the bin count for each generated number.
//! The kernel uses the Philox engine to generate random numbers.
struct RandomInitKernelUniform
{
    //! The kernel operator uses the Philox engine to generate random numbers.
    //! @tparam acc is the accelerator
    //! @param outBins is the output buffer for the histogram of random numbers
    //! @param size is the number of elements to process
    ALPAKA_FN_ACC void operator()(auto const& acc, auto outBins, uint32_t size) const
    {
        // printf("Total number of elements: %u\n", totalFrameExtens);
        //  Each thread is generating it's own seed
        for(auto [seed] :
            alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::onAcc::range::threadsInGrid))
        // Initialize the Philox engine with the global frame element ID as the seed
        {
            // Philox generates uniform integer random numbers
            alpaka::rand::engine::Philox4x32x10 engine(static_cast<uint32_t>(seed));

            auto seedVec = alpaka::Vec{static_cast<uint32_t>(seed)};
            // Define workgroup
            constexpr auto tInGrid = alpaka::onAcc::range::threadsInGrid;
            auto numThreadGroups = tInGrid.getIdxRange(acc).distance();
            auto workGroup = alpaka::onAcc::WorkerGroup{seedVec, numThreadGroups};

            // Iterate over the workgroup
            for([[maybe_unused]] auto [index] : alpaka::onAcc::makeIdxMap(acc, workGroup, alpaka::IdxRange{size}))
            {
                // Generate numbers in [0,1). Philox engine already generates uniform random numbers but they are
                // integers.
                alpaka::rand::distribution::UniformReal<float> dist;
                // get random numbers floating for any vector size

                auto const val = dist(engine);

                // Calculate the bin index for the histogram
                // Since the values are in [0,1), we multiply by the number of bins to get the bin index
                // The bin index is in the range [0, numberOfBins)
                uint32_t binIndex = val * numberOfBins;
                // Atomically increment the bin count in the histogram
                alpaka::onAcc::atomicAdd(acc, &outBins[binIndex], uint32_t{1});
            }
        }
    }
};

// Kernel to generate random numbers and fill the histogram
// Resulting random numbers are collected in bins to generate the histogram
struct RandomInitKernel
{
    //! The kernel operator uses the Philox engine to generate uniform 32-bit integer random numbers.
    //! @tparam acc is the accelerator
    //! @param outBins is the output buffer for the histogram of random numbers
    //! @param size is the number of elements to process
    ALPAKA_FN_ACC void operator()(auto const& acc, auto outBins, uint32_t size) const
    {
        //  Each thread is generating it's own seed
        for(auto [seed] :
            alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::onAcc::range::threadsInGrid))
        // Initialize the Philox engine with the global frame element ID as the seed
        {
            alpaka::rand::engine::Philox4x32x10 engine(static_cast<uint32_t>(seed));

            auto seedVec = alpaka::Vec{static_cast<uint32_t>(seed)};
            // Define workgroup
            constexpr auto tInGrid = alpaka::onAcc::range::threadsInGrid;
            auto numThreadGroups = tInGrid.getIdxRange(acc).distance();
            auto workGroup = alpaka::onAcc::WorkerGroup{seedVec, numThreadGroups};


            // Iterate over the workgroup
            for([[maybe_unused]] auto [index] : alpaka::onAcc::makeIdxMap(acc, workGroup, alpaka::IdxRange{size}))
            {
                // Generate a random 32-bit unsigned integer
                uint32_t val = engine();
                // Calculate the bin index for the histogram
                auto binIndex = (val % numberOfBins);

                // Atomically increment the bin count in the histogram
                alpaka::onAcc::atomicAdd(acc, &outBins[binIndex], uint32_t{1});
            }
        }
    }
};

// Kernel to generate random numbers and fill the histogram
// This kernel uses a vectorized approach to generate 4 integer random numbers at once.
// Results are collected in bins to generate the histogram
struct RandomInitKernelVec
{
    //! The kernel operator uses the Philox engine to generate random numbers.
    //! @tparam acc is the accelerator
    //! @param outBins is the output buffer for the histogram of random numbers
    //! @param size is the number of elements to process
    ALPAKA_FN_ACC void operator()(auto const& acc, auto outBins, uint32_t size) const
    {
        // Generate random values for each element in the buffer.
        for(auto [index] :
            alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{size}))
        {
            // Calculate the engine index
            uint32_t engineIndex = index / 4;

            // Setup generator and distribution.
            alpaka::rand::engine::Philox4x32x10Vector engine(engineIndex);

            // Generate a vector of 4 random numbers, 32 bit integers
            auto valVec = engine();

            // Calculate the offset within the vector
            uint32_t offset = index % 4;

            auto val = valVec[offset];
            // Calculate the bin index for the histogram
            auto binIndex = static_cast<uint32_t>(val % numberOfBins);

            // Atomically increment the bin count in the histogram
            alpaka::onAcc::atomicAdd(acc, &outBins[binIndex], uint32_t{1});
        }
    }
};

bool testRandomInitKernels(
    alpaka::onHost::concepts::Device auto host,
    alpaka::onHost::concepts::Device auto device,
    auto computeExec,
    uint32_t numElements)
{
    using T_Data = uint32_t;
    // Buffer size
    uint32_t const size = numElements;
    constexpr uint32_t numBins = numberOfBins;

    // Allocate input and output host buffers in pinned memory accessible by the Platform devices
    auto outBins_h = alpaka::onHost::alloc<T_Data>(host, numBins);

    // Fill the histogram buffer with zeros
    for(uint32_t i = 0; i < numBins; ++i)
    {
        outBins_h[i] = 0;
    }

    // Run the test on the given device
    alpaka::onHost::Queue queue = device.makeQueue();

    // Allocate output buffer on the device
    auto outBins_d = alpaka::onHost::allocLike(device, outBins_h);

    // Copy the initial data to the device
    alpaka::onHost::memcpy(queue, outBins_d, outBins_h);

    // Frame size

    // Launch the 1-dimensional kernel with scalar size
    auto frameSpec = alpaka::onHost::getFrameSpec(device, computeExec, alpaka::Vec{blockSize});

    // TEST - 1: Philox Generator generates integer random numbers
    std::cout << "- Testing RandomInitKernel with a grid of " << frameSpec << "\n";
    queue.enqueue(frameSpec, RandomInitKernel{}, outBins_d.getMdSpan(), size);

    // Copy the results from the device to the host
    alpaka::onHost::memcpy(queue, outBins_h, outBins_d);

    // Wait for all the operations to complete
    alpaka::onHost::wait(queue);

    // Validate RandomInitKernel Results
    std::vector<T_Data> randomInitHist(std::data(outBins_h), std::data(outBins_h) + numBins);

    // Define a lambda to perform validation and printing
    auto validateResult = [&](char const* kernelName, auto& histogram)
    {
        std::cout << "Validating " << kernelName << " results:\n";

        // Validate the histogram
        uint32_t expectedValue = size / numBins;
        bool const isHistogramValid = validateHistogram(histogram, expectedValue);

        if(isHistogramValid)
        {
            std::cout << "Histogram Validation passed: Histogram distribution is within the expected tolerance.\n";
        }
        else
        {
            std::cerr << "Histogram Validation failed: Histogram distribution is not within the expected tolerance.\n";
        }

        return isHistogramValid;
    };

    // Validate and print results for RandomInitKernel
    bool isValid = validateResult("RandomInitKernel", randomInitHist);

    // TEST - 2 Test RandomInitKernelUniform which generates real random numbers in [0,1)
    // Fill the histogram buffer with zeros
    for(uint32_t i = 0; i < numBins; ++i)
    {
        outBins_h[i] = 0;
    }

    // Copy the initial data to the device
    alpaka::onHost::memcpy(queue, outBins_d, outBins_h);

    std::cout << "- Testing RandomInitKernelUniform with a grid of " << frameSpec << "\n";
    queue.enqueue(frameSpec, RandomInitKernelUniform{}, outBins_d.getMdSpan(), size);

    // Wait for all the operations to complete
    alpaka::onHost::wait(queue);

    // Copy the results from the device to the host
    alpaka::onHost::memcpy(queue, outBins_h, outBins_d);

    // Wait for all the operations to complete
    alpaka::onHost::wait(queue);

    // Validate RandomInitKernelUniform Results
    std::vector<T_Data> randomInitUniformHist(std::data(outBins_h), std::data(outBins_h) + numBins);

    // Validate and print results for RandomInitKernelUniform
    isValid &= validateResult("RandomInitKernelUniform", randomInitUniformHist);

    // TEST - 3 Test RandomInitKernelVec which generates integer random numbers
    // Fill the histogram buffer with zeros
    for(uint32_t i = 0; i < numBins; ++i)
    {
        outBins_h[i] = 0;
    }

    // Copy the initial data to the device
    alpaka::onHost::memcpy(queue, outBins_d, outBins_h);

    std::cout << "- Testing RandomInitKernelVec with a grid of " << frameSpec << "\n";
    queue.enqueue(frameSpec, RandomInitKernelVec{}, outBins_d.getMdSpan(), size);

    // Wait for all the operations to complete
    alpaka::onHost::wait(queue);

    // Copy the results from the device to the host
    alpaka::onHost::memcpy(queue, outBins_h, outBins_d);

    // Wait for all the operations to complete
    alpaka::onHost::wait(queue);

    // Validate Results
    std::vector randomInitVecHist(std::data(outBins_h), std::data(outBins_h) + numBins);

    // Validate and print results
    isValid &= validateResult("RandomInitKernelVec", randomInitVecHist);

    return isValid;
}

int exampleUniformDist(auto const cfg, size_t numElements)
{
    using namespace alpaka;

    auto deviceSpec = alpaka::onHost::DeviceSpec{cfg};
    auto computeExec = alpaka::getExecutor(cfg);


    // Use the single host device
    auto hostSelector = api::host + deviceKind::cpu;
    onHost::Device host = hostSelector.makeDevice(0);
    std::cout << "\n Host:   " << getName(host) << "\n";

    // require at least one device
    std::size_t n = hostSelector.getDeviceCount();

    if(n == 0)
    {
        return EXIT_FAILURE;
    }
    auto deviceSelector = onHost::makeDeviceSelector(deviceSpec);


    // Use the first device
    onHost::Device device = deviceSelector.makeDevice(0);
    std::cout << "Device: " << onHost::getName(device) << "\n";

    bool resultIsCorrect = testRandomInitKernels(host, device, computeExec, numElements);

    if(resultIsCorrect)
    {
        std::cout << "Execution results correct with a certain confidence!" << std::endl;
        return EXIT_SUCCESS;
    }
    else
    {
        std::cout
            << "Execution results incorrect: Random numbers do not follow the expected distribution characteristics."
            << std::endl;
        return EXIT_FAILURE;
    }
}

void help(char* argv[])
{
    std::cerr << argv[0] << " [OPTIONS]" << std::endl;
    std::cerr << "  -n  numElements: Number of elements to process. Default:" << numberOfElements << std::endl;
    std::cerr << "  -e: disable execution of the native std::for_each implementation" << std::endl;
    std::cerr << "  -h: Print this help message" << std::endl;
}

auto main(int argc, char* argv[]) -> int
{
    // Default value if no command line argument used
    size_t numElements = numberOfElements;

    int opt;
    while((opt = getopt(argc, argv, "hn:e")) != -1)
    {
        switch(opt)
        {
        case 'n':
            try
            {
                numElements = std::stoul(optarg, nullptr, 0);
            }
            catch(std::invalid_argument const&)
            {
                std::cerr << "Error: invalid argument '" << optarg << "'.\n";
                return EXIT_FAILURE;
            }
            catch(std::out_of_range const&)
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
    // Execute the example once for each enabled API and executor.
    return onHost::executeForEach(
        [=](alpaka::concepts::BackendSpec auto const& backend)
        {
            auto selector = onHost::makeDeviceSelector(alpaka::onHost::DeviceSpec{backend});
            if(!selector.isAvailable())
                return EXIT_SUCCESS;
            bool retVal = exampleUniformDist(backend, numElements) || exampleNormalDist(backend, numElementsNormal);
            return retVal == false ? EXIT_SUCCESS : EXIT_FAILURE;
        },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
