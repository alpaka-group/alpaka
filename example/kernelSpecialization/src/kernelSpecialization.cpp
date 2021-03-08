/* Copyright 2021 Sergei Bastrakov
 *
 * This file exemplifies usage of alpaka.
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED “AS IS” AND ISC DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL ISC BE LIABLE FOR ANY
 * SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR
 * IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <iostream>

//#############################################################################
//! Kernel to illustrate specialization for a particular accelerator
//!
//! It has a general operator() implementation and a specialized one for the CUDA accelerator.
//! When running the kernel on a CUDA device, the specialized version of operator() is called.
//! Otherwise the general version is called.
//! The same technique can be applied for any function called from inside the kernel,
//! thus allowing specialization of only relevant part of the code.
//! It can be useful for optimization or accessing specific functionality not abstracted by alpaka.
struct Kernel
{
    //-----------------------------------------------------------------------------
    //! Implementation for the general case
    //!
    //! It will be called when no specialization is a better fit.
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc) const
    {
        // For simplicity assume 1d thread indexing
        auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        if(globalThreadIdx == 0u)
            printf("Running the general kernel implementation\n");
    }

    //! Simple partial specialization for the CUDA accelerator
    //!
    //! We have to guard it with #ifdef as the types of alpaka accelerators are only conditionally available.
    //! Specialization for other accelerators is similar, with another template name instead of AccGpuCudaRt.
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    template<typename TDim, typename TIdx>
    ALPAKA_FN_ACC auto operator()(alpaka::AccGpuCudaRt<TDim, TIdx> const& acc) const
    {
        // This specialization is used when the kernel is run on the CUDA accelerator.
        // So inside we can use both alpaka and native CUDA directly.
        // For simplicity assume 1d thread indexing
        auto const globalThreadIdx = blockIdx.x * gridDim.x + threadIdx.x;
        if(globalThreadIdx == 0)
            printf("Running the specialization for the CUDA accelerator\n");
    }
#endif
};

auto main() -> int
{
// Fallback for the CI with disabled sequential backend
#if defined(ALPAKA_CI) && !defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
    return EXIT_SUCCESS;
#else

    // Define the accelerator
    //
    // It is possible to choose from a set of accelerators:
    // - AccGpuCudaRt
    // - AccGpuHipRt
    // - AccCpuThreads
    // - AccCpuFibers
    // - AccCpuOmp2Threads
    // - AccCpuOmp2Blocks
    // - AccOmp5
    // - AccCpuTbbBlocks
    // - AccCpuSerial
    //
    // For simplicity this examples always uses 1 dimensional indexing, and index type size_t
    using Acc = alpaka::ExampleDefaultAcc<alpaka::DimInt<1>, std::size_t>;
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;

    // Defines the synchronization behavior of a queue
    using QueueProperty = alpaka::Blocking;
    using Queue = alpaka::Queue<Acc, QueueProperty>;

    // Select a device
    auto const devAcc = alpaka::getDevByIdx<Acc>(0u);

    // Create a queue on the device
    Queue queue(devAcc);

    // Define the work division
    std::size_t const threadsPerGrid = 16u;
    std::size_t const elementsPerThread = 1u;
    auto const workDiv = alpaka::getValidWorkDiv<Acc>(
        devAcc,
        threadsPerGrid,
        elementsPerThread,
        false,
        alpaka::GridBlockExtentSubDivRestrictions::Unrestricted);

    // Run the kernel
    alpaka::exec<Acc>(queue, workDiv, Kernel{});
    alpaka::wait(queue);

    return EXIT_SUCCESS;
#endif
}
