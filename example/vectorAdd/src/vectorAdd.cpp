/* Copyright 2019 Benjamin Worpitz, Matthias Werner
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

#include <chrono>
#include <iostream>
#include <random>
#include <typeinfo>

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
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc, typename TElem, typename TIdx>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        TElem const* const a,
        TElem const* const b,
        TElem* const c,
        TIdx const& num_elements) const -> void
    {
        static_assert(alpaka::Dim<TAcc>::value == 1, "The VectorAddKernel expects 1-dimensional indices!");

        TIdx const grid_thread_idx(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
        TIdx const thread_elem_extent(alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);
        TIdx const thread_first_elem_idx(grid_thread_idx * thread_elem_extent);

        if(thread_first_elem_idx < num_elements)
        {
            // Calculate the number of elements to compute in this thread.
            // The result is uniform for all but the last thread.
            TIdx const thread_last_elem_idx(thread_first_elem_idx + thread_elem_extent);
            TIdx const thread_last_elem_idx_clipped((num_elements > thread_last_elem_idx) ? thread_last_elem_idx : num_elements);

            for(TIdx i(thread_first_elem_idx); i < thread_last_elem_idx_clipped; ++i)
            {
                c[i] = a[i] + b[i];
            }
        }
    }
};

auto main() -> int
{
// Fallback for the CI with disabled sequential backend
#if defined(ALPAKA_CI) && !defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
    return EXIT_SUCCESS;
#else

    // Define the index domain
    using Dim = alpaka::DimInt<1u>;
    using Idx = std::size_t;

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
    // using Acc = alpaka::AccCpuSerial<Dim, Idx>;
    using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;

    // Defines the synchronization behavior of a queue
    //
    // choose between Blocking and NonBlocking
    using QueueProperty = alpaka::Blocking;
    using QueueAcc = alpaka::Queue<Acc, QueueProperty>;

    // Select a device
    auto const dev_acc = alpaka::getDevByIdx<Acc>(0u);

    // Create a queue on the device
    QueueAcc queue(dev_acc);

    // Define the work division
    Idx const num_elements(123456);
    Idx const elements_per_thread(8u);
    alpaka::Vec<Dim, Idx> const extent(num_elements);

    // Let alpaka calculate good block and grid sizes given our full problem extent
    alpaka::WorkDivMembers<Dim, Idx> const work_div(alpaka::getValidWorkDiv<Acc>(
        dev_acc,
        extent,
        elements_per_thread,
        false,
        alpaka::GridBlockExtentSubDivRestrictions::Unrestricted));

    // Define the buffer element type
    using Data = std::uint32_t;

    // Get the host device for allocating memory on the host.
    using DevHost = alpaka::DevCpu;
    auto const dev_host = alpaka::getDevByIdx<DevHost>(0u);

    // Allocate 3 host memory buffers
    using BufHost = alpaka::Buf<DevHost, Data, Dim, Idx>;
    BufHost buf_host_a(alpaka::allocBuf<Data, Idx>(dev_host, extent));
    BufHost buf_host_b(alpaka::allocBuf<Data, Idx>(dev_host, extent));
    BufHost buf_host_c(alpaka::allocBuf<Data, Idx>(dev_host, extent));

    // Initialize the host input vectors A and B
    Data* const p_buf_host_a(alpaka::getPtrNative(buf_host_a));
    Data* const p_buf_host_b(alpaka::getPtrNative(buf_host_b));
    Data* const p_buf_host_c(alpaka::getPtrNative(buf_host_c));

    // C++14 random generator for uniformly distributed numbers in {1,..,42}
    std::random_device rd{};
    std::default_random_engine eng{rd()};
    std::uniform_int_distribution<Data> dist(1, 42);

    for(Idx i(0); i < num_elements; ++i)
    {
        p_buf_host_a[i] = dist(eng);
        p_buf_host_b[i] = dist(eng);
        p_buf_host_c[i] = 0;
    }

    // Allocate 3 buffers on the accelerator
    using BufAcc = alpaka::Buf<Acc, Data, Dim, Idx>;
    BufAcc buf_acc_a(alpaka::allocBuf<Data, Idx>(dev_acc, extent));
    BufAcc buf_acc_b(alpaka::allocBuf<Data, Idx>(dev_acc, extent));
    BufAcc buf_acc_c(alpaka::allocBuf<Data, Idx>(dev_acc, extent));

    // Copy Host -> Acc
    alpaka::memcpy(queue, buf_acc_a, buf_host_a, extent);
    alpaka::memcpy(queue, buf_acc_b, buf_host_b, extent);
    alpaka::memcpy(queue, buf_acc_c, buf_host_c, extent);

    // Instantiate the kernel function object
    VectorAddKernel kernel;

    // Create the kernel execution task.
    auto const task_kernel = alpaka::createTaskKernel<Acc>(
        work_div,
        kernel,
        alpaka::getPtrNative(buf_acc_a),
        alpaka::getPtrNative(buf_acc_b),
        alpaka::getPtrNative(buf_acc_c),
        num_elements);

    // Enqueue the kernel execution task
    {
        const auto begin_t = std::chrono::high_resolution_clock::now();
        alpaka::enqueue(queue, task_kernel);
        alpaka::wait(queue); // wait in case we are using an asynchronous queue to time actual kernel runtime
        const auto end_t = std::chrono::high_resolution_clock::now();
        std::cout << "Time for kernel execution: " << std::chrono::duration<double>(end_t - begin_t).count() << 's'
                  << std::endl;
    }

    // Copy back the result
    {
        auto begin_t = std::chrono::high_resolution_clock::now();
        alpaka::memcpy(queue, buf_host_c, buf_acc_c, extent);
        alpaka::wait(queue);
        const auto end_t = std::chrono::high_resolution_clock::now();
        std::cout << "Time for HtoD copy: " << std::chrono::duration<double>(end_t - begin_t).count() << 's'
                  << std::endl;
    }

    int false_results = 0;
    static constexpr int max_print_false_results = 20;
    for(Idx i(0u); i < num_elements; ++i)
    {
        Data const& val(p_buf_host_c[i]);
        Data const correct_result(p_buf_host_a[i] + p_buf_host_b[i]);
        if(val != correct_result)
        {
            if(false_results < max_print_false_results)
                std::cerr << "C[" << i << "] == " << val << " != " << correct_result << std::endl;
            ++false_results;
        }
    }

    if(false_results == 0)
    {
        std::cout << "Execution results correct!" << std::endl;
        return EXIT_SUCCESS;
    }
    else
    {
        std::cout << "Found " << false_results << " false results, printed no more than " << max_print_false_results
                  << "\n"
                  << "Execution results incorrect!" << std::endl;
        return EXIT_FAILURE;
    }
#endif
}
