/* Copyright 2025 Mykhailo Varvarin, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExecuteForEachAccTag.hpp>

#include <cstdint>
#include <iostream>

//! Hello world kernel, utilizing grid synchronization.
//! Prints hello world from a thread, performs grid sync.
//! and prints the sum of indixes of this thread and the opposite thread (the sums have to be the same).
//! Prints an error if sum is incorrect.
struct HelloWorldKernel
{
    template<typename Acc>
    ALPAKA_FN_ACC void operator()(Acc const& acc, size_t* array, bool* success) const
    {
        // Get index of the current thread in the grid and the total number of threads.
        size_t gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        size_t gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];

        if(alpaka::oncePerGrid(acc))
        {
            printf("Hello, World from alpaka thread %lu!\n", gridThreadIdx);
        }

        // Write the index of the thread to array.
        array[gridThreadIdx] = gridThreadIdx;

        // Perform grid synchronization.
        alpaka::syncGridThreads(acc);

        // Get the index of the thread from the opposite side of 1D array.
        size_t gridThreadIdxOpposite = array[gridThreadExtent - gridThreadIdx - 1];

        // Sum them.
        size_t sum = gridThreadIdx + gridThreadIdxOpposite;

        // Get the expected sum.
        size_t expectedSum = gridThreadExtent - 1;

        // Print the result and signify an error if the grid synchronization fails.
        if(sum != expectedSum)
        {
            *success = false;
            printf(
                "After grid sync, this thread is %lu, thread on the opposite side is %lu. Their sum is %lu, expected: "
                "%lu.%s",
                gridThreadIdx,
                gridThreadIdxOpposite,
                sum,
                expectedSum,
                sum == expectedSum ? "\n" : " ERROR: the sum is incorrect.\n");
        }
    }
};

// In standard projects, you typically do not execute the code with any available accelerator.
// Instead, a single accelerator is selected once from the active accelerators and the kernels are executed with the
// selected accelerator only. If you use the example as the starting point for your project, you can rename the
// example() function to main() and move the accelerator tag to the function body.
template<typename TAccTag>
auto example(TAccTag const&) -> int
{
    // Define the accelerator
    // For simplicity this examples always uses 1 dimensional indexing, and index type size_t
    using Acc = alpaka::TagToAcc<TAccTag, alpaka::DimInt<1>, std::size_t>;
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;

    // Define dimensionality and type of indices to be used in kernels
    using Dim = alpaka::DimInt<1>;
    using Idx = std::size_t;
    using Scalar = alpaka::Vec<alpaka::DimInt<0u>, Idx>;

    // Select the first device available on a system, for the chosen accelerator
    auto const platform = alpaka::Platform<Acc>{};
    auto const device = getDevByIdx(platform, 0u);

    // Select CPU host
    constexpr auto platformHost = alpaka::Platform<alpaka::DevCpu>{};
    auto const host = getDevByIdx(platformHost, 0u);

    // Define type for a queue with requested properties: Blocking.
    using Queue = alpaka::Queue<Acc, alpaka::Blocking>;
    // Create a queue for the device.
    auto queue = Queue{device};

    // Instantiate the kernel object.
    HelloWorldKernel helloWorldKernel{};

    // Query a valid configuration of blocks, threads per block, and elements per thread for the helloWorldKernel
    // kernel.
    const Idx totalSize = 1024 * 1024;
    const Idx elementsPerThread = 1;
    alpaka::KernelCfg<Acc> config
        = {totalSize, elementsPerThread, false, alpaka::GridBlockExtentSubDivRestrictions::Unrestricted};
    auto grid = alpaka::getValidWorkDiv(config, device, helloWorldKernel, (size_t*) nullptr, (bool*) nullptr);

    // Query the maximum number of active blocks supported by the device for the helloWorldKernel kernel.
    const Idx activeBlocks = static_cast<Idx>(alpaka::getMaxActiveBlocks<Acc>(
        device,
        helloWorldKernel,
        grid.m_blockThreadExtent,
        grid.m_threadElemExtent,
        (size_t*) nullptr,
        (bool*) nullptr));
    std::cout << "Maximum blocks for the kernel: " << activeBlocks << std::endl;

    // Limit the number of blocks to the value supported by the device.
    if(grid.m_gridBlockExtent[0] > activeBlocks)
    {
        grid.m_gridBlockExtent[0] = activeBlocks;
    }

    // Allocate memory on the device.
    auto d_array = alpaka::allocBuf<size_t, Idx>(device, Idx{grid.m_gridBlockExtent[0] * grid.m_blockThreadExtent[0]});

    // Allocate the result value.
    auto d_result = alpaka::allocBuf<bool, Idx>(device, Scalar{});
    alpaka::memset(queue, d_result, static_cast<std::uint8_t>(true));

    // Create a task to run the kernel.
    // Note the cooperative kernel specification: only cooperative kernels can
    // perform grid synchronization.
    auto taskRunKernel
        = alpaka::createTaskCooperativeKernel<Acc>(grid, helloWorldKernel, d_array.data(), d_result.data());

    // Enqueue the kernel execution task.
    alpaka::enqueue(queue, taskRunKernel);

    // Copy the result value to the host
    auto h_result = alpaka::allocBuf<bool, Idx>(host, Scalar{});
    alpaka::memcpy(queue, h_result, d_result);
    alpaka::wait(queue);

    return *h_result ? EXIT_SUCCESS : EXIT_FAILURE;
}

auto main() -> int
{
    // Execute the example once for each enabled accelerator.
    // If you would like to execute it for a single accelerator only you can use the following code.
    //  \code{.cpp}
    //  auto tag = TagCpuSerial;
    //  return example(tag);
    //  \endcode
    //
    // valid tags:
    //   TagCpuSerial, TagGpuHipRt, TagGpuCudaRt, TagCpuOmp2Blocks, TagCpuTbbBlocks,
    //   TagCpuOmp2Threads, TagCpuSycl, TagCpuTbbBlocks, TagCpuThreads,
    //   TagFpgaSyclIntel, TagGenericSycl, TagGpuSyclIntel
    return alpaka::executeForEachAccTag([=](auto const& tag) { return example(tag); });
}
