/* Copyright 2024 Mykhailo Varvarin
* SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <iostream>

//! Hello world kernel, utilizing grid synchronization.
//! Prints hello world from a thread, performs grid sync.
//! and prints the sum of indixes of this thread and the opposite thread (the sums have to be the same).
//! Prints an error if sum is incorrect.
struct HelloWorldKernel
{
    template<typename Acc>
    ALPAKA_FN_ACC void operator()(Acc const& acc, uint32_t* data) const
    {
        // Get index of the current thread in the grid and the total number of threads.
        uint32_t gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];;
        uint32_t gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];

        printf("Hello, World from alpaka thread %u!\n", gridThreadIdx);

        // Write the index of the thread to array.
        data[gridThreadIdx] = gridThreadIdx;

        // Perform grid synchronization.
        alpaka::syncGridThreads(acc);

        // Get the index of the opposite thread.
        uint32_t gridThreadIdxOpposite = data[gridThreadExtent - gridThreadIdx - 1];

        // Sum them.
        uint32_t sum = gridThreadIdx + gridThreadIdxOpposite;

        // Get the expected sum.
        uint32_t expectedSum = gridThreadExtent - 1;

        // Print the result and signify an error if the grid synchronization fails.
        printf(
            "After grid sync, this thread is %u, thread on the opposite side is %u. Their sum is %u, expected: %u.%s",
            gridThreadIdx,
            gridThreadIdxOpposite,
            sum,
            expectedSum,
            sum == expectedSum ? "\n" : " ERROR: the sum is incorrect.\n");
    }
};

auto main() -> int
{
    // Define dimensionality and type of indices to be used in kernels
    using Dim = alpaka::DimInt<1>;
    using Idx = uint32_t;

    // Define alpaka accelerator type, which corresponds to the underlying programming model
    using Acc = alpaka::AccGpuCudaRt<Dim, Idx>;

    // Select the first device available on a system, for the chosen accelerator
    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const devAcc = getDevByIdx(platformAcc, 0u);

    // Define type for a queue with requested properties: Blocking.
    using Queue = alpaka::Queue<Acc, alpaka::Blocking>;
    // Create a queue for the device.
    auto queue = Queue{devAcc};

    // Define kernel execution configuration of blocks,
    // threads per block, and elements per thread.
    Idx blocksPerGrid = 10;
    Idx threadsPerBlock = 1;
    Idx elementsPerThread = 1;
    using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;
    auto workDiv = WorkDiv{blocksPerGrid, threadsPerBlock, elementsPerThread};

    // Allocate memory on the device.
    alpaka::Vec<Dim, Idx> bufferExtent{blocksPerGrid * threadsPerBlock};
    auto deviceMemory = alpaka::allocBuf<uint32_t, Idx>(devAcc, bufferExtent);

    // Instantiate the kernel object.
    HelloWorldKernel helloWorldKernel;

    // Create a task to run the kernel.
    // Note the cooperative kernel specification.
    // Only cooperative kernels can perform grid synchronization.
    auto taskRunKernel = alpaka::createTaskCooperativeKernel<Acc>(workDiv, helloWorldKernel, getPtrNative(deviceMemory));

    // Enqueue the kernel execution task..
    alpaka::enqueue(queue, taskRunKernel);
    return 0;
}
