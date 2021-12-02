/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "mysqrt.hpp"

#include <alpaka/test/MeasureKernelRunTime.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch.hpp>

#include <iostream>
#include <typeinfo>

//! A vector addition kernel.
class SqrtKernel
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

        auto const grid_thread_idx(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
        auto const thread_elem_extent(alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);
        auto const thread_first_elem_idx(grid_thread_idx * thread_elem_extent);

        if(thread_first_elem_idx < num_elements)
        {
            // Calculate the number of elements to compute in this thread.
            // The result is uniform for all but the last thread.
            auto const thread_last_elem_idx(thread_first_elem_idx + thread_elem_extent);
            auto const thread_last_elem_idx_clipped((num_elements > thread_last_elem_idx) ? thread_last_elem_idx : num_elements);

            for(TIdx i(thread_first_elem_idx); i < thread_last_elem_idx_clipped; ++i)
            {
                c[i] = mysqrt(a[i]) + mysqrt(b[i]);
            }
        }
    }
};

using TestAccs = alpaka::test::EnabledAccs<alpaka::DimInt<1u>, std::size_t>;

TEMPLATE_LIST_TEST_CASE("separableCompilation", "[separableCompilation]", TestAccs)
{
    using Acc = TestType;
    using Idx = alpaka::Idx<Acc>;

    using Val = double;

    using DevAcc = alpaka::Dev<Acc>;
    using PltfAcc = alpaka::Pltf<DevAcc>;
    using QueueAcc = alpaka::test::DefaultQueue<alpaka::Dev<Acc>>;
    using PltfHost = alpaka::PltfCpu;
    using DevHost = alpaka::Dev<PltfHost>;

    Idx const num_elements(32);

    // Create the kernel function object.
    SqrtKernel kernel;

    // Get the host device.
    DevHost const dev_host = alpaka::getDevByIdx<PltfHost>(0u);

    // Select a device to execute on.
    DevAcc const dev_acc = alpaka::getDevByIdx<PltfAcc>(0);

    // Get a queue on this device.
    QueueAcc queue_acc(dev_acc);

    // The data extent.
    alpaka::Vec<alpaka::DimInt<1u>, Idx> const extent(num_elements);

    // Let alpaka calculate good block and grid sizes given our full problem extent.
    alpaka::WorkDivMembers<alpaka::DimInt<1u>, Idx> const work_div(alpaka::getValidWorkDiv<Acc>(
        dev_acc,
        extent,
        static_cast<Idx>(3u),
        false,
        alpaka::GridBlockExtentSubDivRestrictions::Unrestricted));

    std::cout << typeid(kernel).name() << "("
              << "accelerator: " << alpaka::getAccName<Acc>() << ", workDiv: " << work_div
              << ", numElements:" << num_elements << ")" << std::endl;

    // Allocate host memory buffers.
    auto mem_buf_host_a(alpaka::allocBuf<Val, Idx>(dev_host, extent));
    auto mem_buf_host_b(alpaka::allocBuf<Val, Idx>(dev_host, extent));
    auto mem_buf_host_c(alpaka::allocBuf<Val, Idx>(dev_host, extent));

    // Initialize the host input vectors
    for(Idx i(0); i < num_elements; ++i)
    {
        alpaka::getPtrNative(mem_buf_host_a)[i] = static_cast<Val>(rand()) / static_cast<Val>(RAND_MAX);
        alpaka::getPtrNative(mem_buf_host_b)[i] = static_cast<Val>(rand()) / static_cast<Val>(RAND_MAX);
    }

    // Allocate the buffers on the accelerator.
    auto mem_buf_acc_a(alpaka::allocBuf<Val, Idx>(dev_acc, extent));
    auto mem_buf_acc_b(alpaka::allocBuf<Val, Idx>(dev_acc, extent));
    auto mem_buf_acc_c(alpaka::allocBuf<Val, Idx>(dev_acc, extent));

    // Copy Host -> Acc.
    alpaka::memcpy(queue_acc, mem_buf_acc_a, mem_buf_host_a, extent);
    alpaka::memcpy(queue_acc, mem_buf_acc_b, mem_buf_host_b, extent);

    // Create the executor task.
    auto const task_kernel = alpaka::createTaskKernel<Acc>(
        work_div,
        kernel,
        alpaka::getPtrNative(mem_buf_acc_a),
        alpaka::getPtrNative(mem_buf_acc_b),
        alpaka::getPtrNative(mem_buf_acc_c),
        num_elements);

    // Profile the kernel execution.
    std::cout << "Execution time: " << alpaka::test::integ::measureTaskRunTimeMs(queue_acc, task_kernel) << " ms"
              << std::endl;

    // Copy back the result.
    alpaka::memcpy(queue_acc, mem_buf_host_c, mem_buf_acc_c, extent);
    alpaka::wait(queue_acc);

    bool result_correct(true);
    auto const p_host_data(alpaka::getPtrNative(mem_buf_host_c));
    for(Idx i(0u); i < num_elements; ++i)
    {
        auto const& val(p_host_data[i]);
        auto const correct_result(
            std::sqrt(alpaka::getPtrNative(mem_buf_host_a)[i]) + std::sqrt(alpaka::getPtrNative(mem_buf_host_b)[i]));
        auto const abs_diff = (val - correct_result);
        if(abs_diff > std::numeric_limits<Val>::epsilon())
        {
            std::cout << "C[" << i << "] == " << val << " != " << correct_result << std::endl;
            result_correct = false;
        }
    }

    REQUIRE(true == result_correct);
}
