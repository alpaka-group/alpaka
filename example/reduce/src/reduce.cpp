/* Copyright 2019 Benjamin Worpitz, Jonas Schenke, Matthias Werner
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

#include "alpakaConfig.hpp"
#include "kernel.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdlib>
#include <iostream>

// It requires support for extended lambdas when using nvcc as CUDA compiler.
// Requires sequential backend if CI is used
#if(!defined(__NVCC__) || (defined(__NVCC__) && defined(__CUDACC_EXTENDED_LAMBDA__)))                                 \
    && (!defined(ALPAKA_CI) || defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED))

// use defines of a specific accelerator from alpakaConfig.hpp
// that are defined in alpakaConfig.hpp
// - GpuCudaRt
// - CpuThreads
// - CpuOmp2Blocks
// - Omp5
// - CpuSerial
//
using Accelerator = CpuSerial;

using Acc = Accelerator::Acc;
using Host = Accelerator::Host;
using QueueProperty = alpaka::Blocking;
using QueueAcc = alpaka::Queue<Acc, QueueProperty>;
using MaxBlockSize = Accelerator::MaxBlockSize;

//! Reduces the numbers 1 to n.
//!
//! \tparam T The data type.
//! \tparam TFunc The data type of the reduction functor.
//!
//! \param devHost The host device.
//! \param devAcc The accelerator object.
//! \param queue The device queue.
//! \param n The problem size.
//! \param hostMemory The buffer containing the input data.
//! \param func The reduction function.
//!
//! Returns true if the reduction was correct and false otherwise.
template<typename T, typename TDevHost, typename TDevAcc, typename TFunc>
T reduce(
    TDevHost dev_host,
    TDevAcc dev_acc,
    QueueAcc queue,
    uint64_t n,
    alpaka::Buf<TDevHost, T, Dim, Idx> host_memory,
    TFunc func)
{
    static constexpr uint64_t blockSize = getMaxBlockSize<Accelerator, 256>();

    // calculate optimal block size (8 times the MP count proved to be
    // relatively near to peak performance in benchmarks)
    auto block_count = static_cast<uint32_t>(alpaka::getAccDevProps<Acc>(dev_acc).m_multiProcessorCount * 8);
    auto max_block_count = static_cast<uint32_t>((((n + 1) / 2) - 1) / blockSize + 1); // ceil(ceil(n/2.0)/blockSize)

    if(block_count > max_block_count)
        block_count = max_block_count;

    alpaka::Buf<TDevAcc, T, Dim, Extent> source_device_memory = alpaka::allocBuf<T, Idx>(dev_acc, n);

    alpaka::Buf<TDevAcc, T, Dim, Extent> destination_device_memory
        = alpaka::allocBuf<T, Idx>(dev_acc, static_cast<Extent>(block_count));

    // copy the data to the GPU
    alpaka::memcpy(queue, source_device_memory, host_memory, n);

    // create kernels with their workdivs
    ReduceKernel<blockSize, T, TFunc> kernel1;
    ReduceKernel<blockSize, T, TFunc> kernel2;
    WorkDiv work_div1{static_cast<Extent>(block_count), static_cast<Extent>(blockSize), static_cast<Extent>(1)};
    WorkDiv work_div2{static_cast<Extent>(1), static_cast<Extent>(blockSize), static_cast<Extent>(1)};

    // create main reduction kernel execution task
    auto const task_kernel_reduce_main = alpaka::createTaskKernel<Acc>(
        work_div1,
        kernel1,
        alpaka::getPtrNative(source_device_memory),
        alpaka::getPtrNative(destination_device_memory),
        n,
        func);

    // create last block reduction kernel execution task
    auto const task_kernel_reduce_last_block = alpaka::createTaskKernel<Acc>(
        work_div2,
        kernel2,
        alpaka::getPtrNative(destination_device_memory),
        alpaka::getPtrNative(destination_device_memory),
        block_count,
        func);

    // enqueue both kernel execution tasks
    alpaka::enqueue(queue, task_kernel_reduce_main);
    alpaka::enqueue(queue, task_kernel_reduce_last_block);

    //  download result from GPU
    T result_gpu_host;
    auto result_gpu_device = alpaka::createView(dev_host, &result_gpu_host, static_cast<Extent>(blockSize));

    alpaka::memcpy(queue, result_gpu_device, destination_device_memory, 1);

    return result_gpu_host;
}

int main()
{
    // select device and problem size
    const int dev = 0;
    uint64_t n = 1 << 28;

    using T = uint32_t;
    static constexpr uint64_t block_size = getMaxBlockSize<Accelerator, 256>();

    auto dev_acc = alpaka::getDevByIdx<Acc>(dev);
    auto dev_host = alpaka::getDevByIdx<Host>(0u);
    QueueAcc queue(dev_acc);

    // calculate optimal block size (8 times the MP count proved to be
    // relatively near to peak performance in benchmarks)
    uint32_t block_count = static_cast<uint32_t>(alpaka::getAccDevProps<Acc>(dev_acc).m_multiProcessorCount * 8);
    auto max_block_count = static_cast<uint32_t>((((n + 1) / 2) - 1) / block_size + 1); // ceil(ceil(n/2.0)/blockSize)

    if(block_count > max_block_count)
        block_count = max_block_count;

    // allocate memory
    auto host_memory = alpaka::allocBuf<T, Idx>(dev_host, n);

    T* native_host_memory = alpaka::getPtrNative(host_memory);

    // fill array with data
    for(uint64_t i = 0; i < n; i++)
        native_host_memory[i] = static_cast<T>(i + 1);

    // define the reduction function
    auto add_fn = [] ALPAKA_FN_ACC(T a, T b) -> T { return a + b; };

    // reduce
    auto result = reduce<T>(dev_host, dev_acc, queue, n, host_memory, add_fn);
    alpaka::wait(queue);

    // check result
    auto expected_result = static_cast<T>(n / 2 * (n + 1));
    if(result != expected_result)
    {
        std::cerr << "Results don't match: " << result << " != " << expected_result << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "Results match.\n";

    return EXIT_SUCCESS;
}

#else

int main()
{
    return EXIT_SUCCESS;
}

#endif
