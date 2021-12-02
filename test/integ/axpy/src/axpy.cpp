/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/test/MeasureKernelRunTime.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <typeinfo>

//! A vector addition kernel.
class AxpyKernel
{
public:
    //! Vector addition Y = alpha * X + Y.
    //!
    //! \tparam TAcc The type of the accelerator the kernel is executed on..
    //! \tparam TElem The matrix element type.
    //! \param acc The accelerator the kernel is executed on.
    //! \param numElements Specifies the number of elements of the vectors X and Y.
    //! \param alpha Scalar the X vector is multiplied with.
    //! \param X Vector of at least n elements.
    //! \param Y Vector of at least n elements.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc, typename TElem, typename TIdx>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        TIdx const& num_elements,
        TElem const& alpha,
        TElem const* const x,
        TElem* const y) const -> void
    {
        static_assert(alpaka::Dim<TAcc>::value == 1, "The AxpyKernel expects 1-dimensional indices!");

        auto const grid_thread_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];
        auto const thread_elem_extent = alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u];
        auto const thread_first_elem_idx = grid_thread_idx * thread_elem_extent;

        if(thread_first_elem_idx < num_elements)
        {
            // Calculate the number of elements to compute in this thread.
            // The result is uniform for all but the last thread.
            auto const thread_last_elem_idx = thread_first_elem_idx + thread_elem_extent;
            auto const thread_last_elem_idx_clipped = (num_elements > thread_last_elem_idx) ? thread_last_elem_idx : num_elements;

            for(TIdx i(thread_first_elem_idx); i < thread_last_elem_idx_clipped; ++i)
            {
                y[i] = alpha * x[i] + y[i];
            }
        }
    }
};

using TestAccs = alpaka::test::EnabledAccs<alpaka::DimInt<1u>, std::size_t>;

TEMPLATE_LIST_TEST_CASE("axpy", "[axpy]", TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

#ifdef ALPAKA_CI
    Idx const numElements = 1u << 9u;
#else
    Idx const num_elements = 1u << 16u;
#endif

    using Val = float;
    using DevAcc = alpaka::Dev<Acc>;
    using PltfAcc = alpaka::Pltf<DevAcc>;
    using QueueAcc = alpaka::test::DefaultQueue<DevAcc>;
    using PltfHost = alpaka::PltfCpu;

    // Create the kernel function object.
    AxpyKernel kernel;

    // Get the host device.
    auto const dev_host = alpaka::getDevByIdx<PltfHost>(0u);

    // Select a device to execute on.
    auto const dev_acc = alpaka::getDevByIdx<PltfAcc>(0u);

    // Get a queue on this device.
    QueueAcc queue(dev_acc);

    alpaka::Vec<Dim, Idx> const extent(num_elements);

    // Let alpaka calculate good block and grid sizes given our full problem extent.
    alpaka::WorkDivMembers<Dim, Idx> const work_div(alpaka::getValidWorkDiv<Acc>(
        dev_acc,
        extent,
        static_cast<Idx>(3u),
        false,
        alpaka::GridBlockExtentSubDivRestrictions::Unrestricted));

    std::cout << "AxpyKernel("
              << " numElements:" << num_elements << ", accelerator: " << alpaka::getAccName<Acc>()
              << ", kernel: " << typeid(kernel).name() << ", workDiv: " << work_div << ")" << std::endl;

    // Allocate host memory buffers.
    auto mem_buf_host_x = alpaka::allocBuf<Val, Idx>(dev_host, extent);
    auto mem_buf_host_orig_y = alpaka::allocBuf<Val, Idx>(dev_host, extent);
    auto mem_buf_host_y = alpaka::allocBuf<Val, Idx>(dev_host, extent);
    Val* const p_buf_host_x = alpaka::getPtrNative(mem_buf_host_x);
    Val* const p_buf_host_orig_y = alpaka::getPtrNative(mem_buf_host_orig_y);
    Val* const p_buf_host_y = alpaka::getPtrNative(mem_buf_host_y);

    // random generator for uniformly distributed numbers in [0,1)
    // keep in mind, this can generate different values on different platforms
    std::random_device rd{};
    auto const seed = rd();
    std::default_random_engine eng{seed};
    std::uniform_real_distribution<Val> dist(0.0, 1.0);
    std::cout << "using seed: " << seed << "\n";
    // Initialize the host input vectors
    for(Idx i(0); i < num_elements; ++i)
    {
        p_buf_host_x[i] = dist(eng);
        p_buf_host_orig_y[i] = dist(eng);
    }
    Val const alpha(dist(eng));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
    std::cout << __func__ << " alpha: " << alpha << std::endl;
    std::cout << __func__ << " X_host: ";
    alpaka::print(memBufHostX, std::cout);
    std::cout << std::endl;
    std::cout << __func__ << " Y_host: ";
    alpaka::print(memBufHostOrigY, std::cout);
    std::cout << std::endl;
#endif

    // Allocate the buffer on the accelerator.
    auto mem_buf_acc_x = alpaka::allocBuf<Val, Idx>(dev_acc, extent);
    auto mem_buf_acc_y = alpaka::allocBuf<Val, Idx>(dev_acc, extent);

    // Copy Host -> Acc.
    alpaka::memcpy(queue, mem_buf_acc_x, mem_buf_host_x, extent);
    alpaka::memcpy(queue, mem_buf_acc_y, mem_buf_host_orig_y, extent);

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
    alpaka::wait(queue);

    std::cout << __func__ << " X_Dev: ";
    alpaka::print(memBufHostX, std::cout);
    std::cout << std::endl;
    std::cout << __func__ << " Y_Dev: ";
    alpaka::print(memBufHostX, std::cout);
    std::cout << std::endl;
#endif

    // Create the kernel execution task.
    auto const task_kernel = alpaka::createTaskKernel<Acc>(
        work_div,
        kernel,
        num_elements,
        alpha,
        alpaka::getPtrNative(mem_buf_acc_x),
        alpaka::getPtrNative(mem_buf_acc_y));

    // Profile the kernel execution.
    std::cout << "Execution time: " << alpaka::test::integ::measureTaskRunTimeMs(queue, task_kernel) << " ms"
              << std::endl;

    // Copy back the result.
    alpaka::memcpy(queue, mem_buf_host_y, mem_buf_acc_y, extent);

    // Wait for the queue to finish the memory operation.
    alpaka::wait(queue);

    bool result_correct(true);
    for(Idx i(0u); i < num_elements; ++i)
    {
        auto const& val(p_buf_host_y[i]);
        auto const correct_result = alpha * p_buf_host_x[i] + p_buf_host_orig_y[i];
        auto const rel_diff = std::abs((val - correct_result) / std::min(val, correct_result));
        if(rel_diff > std::numeric_limits<Val>::epsilon())
        {
            std::cerr << "C[" << i << "] == " << val << " != " << correct_result << std::endl;
            result_correct = false;
        }
    }

    REQUIRE(result_correct);
}
