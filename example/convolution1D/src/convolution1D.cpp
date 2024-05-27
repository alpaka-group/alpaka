/* Copyright 2023  Bernhard Manfred Gruber, Simeon Ehrig, Rene Widera, Mehmet Yusufoglu.
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <type_traits>

//! Convolution Example
//!
//! 1D convolution example: Creates two 1D arrays, applies convolution filter.
//! Array sizes are hardcoded.
//!

/**
 * @brief The ConvolutionKernel function-object
 * Calculates 1D convolution using input and filter arrays.
 */
struct ConvolutionKernel
{
    /** @brief Main convolution code
     *  @param Accelerator
     *  @param Input array, first input of convolution integral
     *  @param Filter array, second input of convolution integral
     *  @param Empty output array to be filled
     *  @param Input array size
     *  @param Filter size
     */
    template<typename TAcc, typename TElem>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        TElem const* const input,
        TElem const* const filter,
        TElem* const output,
        const std::size_t inputSize,
        const std::size_t filterSize) const -> void
    {
        auto const globalThreadIdxX = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];

        // Since the kernel is launched 1-D calculating linearizedGlobalThreadIdx line is unnecessary.
        // globalThreadIdx[0] can be used to map all the threads.
        if(globalThreadIdxX < inputSize)
        {
            int32_t const halfFilterSize = filterSize / 2;
            TElem result = 0.0f;
            // Calculate sum of multiplications of corresponding elements
            auto const start
                = static_cast<int32_t>(std::max(static_cast<int32_t>(globalThreadIdxX) - halfFilterSize, 0));
            auto const stop = std::min(globalThreadIdxX + halfFilterSize, inputSize - 1);
            for(int32_t i = start; i <= stop; ++i)
                result += input[i] * filter[i + halfFilterSize - static_cast<int32_t>(globalThreadIdxX)];
            output[globalThreadIdxX] = result;
        }
    }
};

auto FuzzyEqual(float a, float b) -> bool
{
    return std::fabs(a - b) < std::numeric_limits<float>::epsilon() * 10.0f;
}

auto main() -> int
{
    // Size of 1D arrays to be used in convolution integral
    // Here instead of "convolution kernel" the term "filter" is used because kernel has a different meaning in GPU
    // programming. Secondly filter array is not reversed. Implemented like a convolutional layer in CNN.
    constexpr size_t filterSize = 3;
    using DataType = float;
    constexpr size_t inputSize = 8;
    constexpr std::array<DataType, inputSize> expectedOutput = {0.8f, 1.4f, 2.0f, 2.6f, 3.2f, 3.8f, 4.4f, 2.3f};

    // Define the index domain
    using Dim = alpaka::DimInt<1u>;
    // Index type
    using Idx = std::size_t;

    // Define the accelerator
    using DevAcc = alpaka::ExampleDefaultAcc<Dim, Idx>;
    using QueueProperty = alpaka::Blocking;
    using QueueAcc = alpaka::Queue<DevAcc, QueueProperty>;
    using BufAcc = alpaka::Buf<DevAcc, DataType, Dim, Idx, alpaka::MemVisibilityTypeList<DevAcc>>;

    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<DevAcc>() << '\n';

    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);

    // Select a device
    auto const platformAcc = alpaka::Platform<DevAcc>{};
    auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);

    // Create a queue on the device
    QueueAcc queue(devAcc);

    // Allocate memory host input
    auto hostInputMemory = alpaka::allocBuf<DataType, Idx>(devHost, inputSize);

    // Fill array with data
    for(size_t i = 0; i < inputSize; i++)
        hostInputMemory[i] = static_cast<DataType>(i + 1);

    // Allocate memory host filter
    auto hostFilterMemory = alpaka::allocBuf<DataType, Idx>(devHost, filterSize);

    // Fill array with any data
    for(size_t i = 0; i < filterSize; i++)
        hostFilterMemory[i] = static_cast<DataType>(i + 1) / 10.0f;

    // Allocate memory in device
    BufAcc inputDeviceMemory = alpaka::allocBuf<DataType, Idx>(devAcc, inputSize);
    BufAcc filterDeviceMemory = alpaka::allocBuf<DataType, Idx>(devAcc, filterSize);
    BufAcc outputDeviceMemory = alpaka::allocBuf<DataType, Idx>(devAcc, static_cast<Idx>(inputSize));

    // Copy input and filter (convolution kernel array) from host to device
    alpaka::memcpy(queue, inputDeviceMemory, hostInputMemory, inputSize);
    alpaka::memcpy(queue, filterDeviceMemory, hostFilterMemory, filterSize);
    // Make sure memcpy finished.
    alpaka::wait(queue);
    using Vec = alpaka::Vec<Dim, Idx>;
    using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;

    auto const elementsPerThread = Vec::all(static_cast<Idx>(1));
    // Grid size
    auto const threadsPerGrid = inputSize;
    WorkDiv const workDiv = alpaka::getValidWorkDiv<DevAcc>(
        devAcc,
        threadsPerGrid,
        elementsPerThread,
        false,
        alpaka::GridBlockExtentSubDivRestrictions::Unrestricted);

    // Instantiate the kernel (gpu code) function-object
    ConvolutionKernel convolutionKernel;

    // Native pointers needed for the kernel execution function
    DataType* nativeFilterDeviceMemory = std::data(filterDeviceMemory);
    DataType* nativeInputDeviceMemory = std::data(inputDeviceMemory);
    DataType* nativeOutputDeviceMemory = std::data(outputDeviceMemory);

    // Run the kernel
    alpaka::exec<DevAcc>(
        queue,
        workDiv,
        convolutionKernel,
        nativeInputDeviceMemory,
        nativeFilterDeviceMemory,
        nativeOutputDeviceMemory,
        inputSize,
        filterSize);

    // Allocate memory on host
    auto resultGpuHost = alpaka::allocBuf<DataType, Idx>(devHost, inputSize);
    // Copy from device memory to host
    alpaka::memcpy(queue, resultGpuHost, outputDeviceMemory, inputSize);
    alpaka::wait(queue);

    bool allEqual{true};
    // Print result array at the host
    for(size_t i{0}; i < inputSize; i++)
    {
        std::cout << "output[" << i << "]:" << std::setprecision(3) << resultGpuHost[i] << "\n";
        // Compare with the reference output
        bool fuzzyEqual = FuzzyEqual(resultGpuHost[i], expectedOutput[i]);
        allEqual = allEqual && fuzzyEqual;
    }
    if(!allEqual)
    {
        std::cout << "Error: Some convolution results doesn't match!\n";
        return EXIT_FAILURE;
    }
    std::cout << "All results are correct!\n";
    return EXIT_SUCCESS;
}
