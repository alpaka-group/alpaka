/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera
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

#include <functional>
#include <iostream>
#include <typeinfo>
#include <vector>

//! A matrix multiplication kernel.
//! Computes C + alpha*A*B + beta*C. LxM * MxN -> LxN
//! This is an adaption of the algorithm from the CUDA developers guide.
class MatMulKernel
{
public:
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \tparam TElem The matrix element type.
    //! \param acc The accelerator to be executed on.
    //! \param m The height of the A matrix.
    //! \param n The width of the A and height of the B matrix.
    //! \param k The width of the B matrix.
    //! \param A The pointer to the matrix A data.
    //! \param lda The pitch of the A matrix in elements.
    //! \param B The pointer to the matrix B data.
    //! \param ldb The pitch of the B matrix in elements.
    //! \param C The pointer to the matrix C data.
    //! \param ldc The pitch of the C matrix in elements.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc, typename TElem, typename TIndex>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        TIndex const& m,
        TIndex const& n,
        TIndex const& k,
        TElem const& alpha,
        TElem const* const a,
        TIndex const& lda,
        TElem const* const b,
        TIndex const& ldb,
        TElem const& beta,
        TElem* const c,
        TIndex const& ldc) const -> void
    {
        static_assert(
            alpaka::Dim<TAcc>::value == 2u,
            "The accelerator used for the GemmAlpakaKernel has to be 2 dimensional!");

        // Column and row of C to calculate.
        auto const grid_thread_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const& grid_thread_idx_x = grid_thread_idx[1u];
        auto const& grid_thread_idx_y = grid_thread_idx[0u];

        // Column and row inside the block of C to calculate.
        auto const block_thread_idx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
        auto const& block_thread_idx_x = block_thread_idx[1u];
        auto const& block_thread_idx_y = block_thread_idx[0u];

        // The block threads extent.
        auto const block_thread_extent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        auto const& block_thread_extent_x = block_thread_extent[1u];
        auto const& block_thread_extent_y = block_thread_extent[0u];
        // ALPAKA_ASSERT(blockThreadExtentX == blockThreadExtentY);
        auto const& block_thread_extent_val = block_thread_extent_x;

        // Shared memory used to store the current blocks of A and B.
        auto* const p_block_shared_a = alpaka::getDynSharedMem<TElem>(acc);
        auto* const p_block_shared_b = p_block_shared_a + block_thread_extent_x * block_thread_extent_y;

        auto const shared_block_idx1d = block_thread_idx_y * block_thread_extent_x + block_thread_idx_x;

        // If the element corresponding to the current thread is outside of the respective matrix.
        bool const inside_a(grid_thread_idx_y < m);
        bool const inside_b(grid_thread_idx_x < n);
        bool const inside_c(inside_a && inside_b);

        TElem dot_product(0);

        // Loop over all blocks of A and B that are required to compute the C block.
        auto const block_mul_count
            = static_cast<TIndex>(std::ceil(static_cast<float>(k) / static_cast<float>(block_thread_extent_val)));
        for(TIndex k2(0u); k2 < block_mul_count; ++k2)
        {
            // Copy the current blocks of A and B into shared memory in parallel.
            // If the element of the current thread is outside of the matrix, zero is written into the shared memory.
            // This is possible because zero is a result neutral extension of the matrices regarding the dot product.
            auto const a_idx_x = k2 * block_thread_extent_x + block_thread_idx_x;
            auto const a_idx1d = grid_thread_idx_y * lda + a_idx_x;
            p_block_shared_a[shared_block_idx1d] = (((!inside_a) || (a_idx_x >= k)) ? static_cast<TElem>(0) : a[a_idx1d]);

            auto const b_idx_y = k2 * block_thread_extent_y + block_thread_idx_y;
            auto const b_idx1d = b_idx_y * ldb + grid_thread_idx_x;
            p_block_shared_b[shared_block_idx1d] = (((!inside_b) || (b_idx_y >= k)) ? static_cast<TElem>(0) : b[b_idx1d]);

            // Synchronize to make sure the complete blocks are loaded before starting the computation.
            alpaka::syncBlockThreads(acc);

            // Not really necessary because we wrote zeros into those cells.
            // if(insideC)
            //{
            // Compute the dot products within shared memory.
            for(TIndex k3(0); k3 < block_thread_extent_val; ++k3)
            {
                dot_product += p_block_shared_a[block_thread_idx_y * block_thread_extent_x + k3]
                    * p_block_shared_b[k3 * block_thread_extent_y + block_thread_idx_x];
            }
            //}

            // Synchronize to make sure that the preceding computation is done before loading the next blocks of A and
            // B.
            alpaka::syncBlockThreads(acc);
        }

        // If the element is outside of the matrix it was only a helper thread that did not calculate any meaningful
        // results.
        if(inside_c)
        {
            auto const c_idx1d = grid_thread_idx_y * ldc + grid_thread_idx_x;
            c[c_idx1d] = alpha * dot_product + beta * c[c_idx1d];
        }
    }
};

namespace alpaka
{
    namespace traits
    {
        //! The trait for getting the size of the block shared dynamic memory for a kernel.
        template<typename TAcc>
        struct BlockSharedMemDynSizeBytes<MatMulKernel, TAcc>
        {
            //! \return The size of the shared memory allocated for a block.
            template<typename TVec, typename TIndex, typename TElem>
            ALPAKA_FN_HOST_ACC static auto get_block_shared_mem_dyn_size_bytes(
                MatMulKernel const& mat_mul_kernel,
                TVec const& block_thread_extent,
                TVec const& thread_elem_extent,
                TIndex const& m,
                TIndex const& n,
                TIndex const& k,
                TElem const& alpha,
                TElem const* const a,
                TIndex const& lda,
                TElem const* const b,
                TIndex const& ldb,
                TElem const& beta,
                TElem* const c,
                TIndex const& ldc)
            {
                alpaka::ignore_unused(mat_mul_kernel);
                alpaka::ignore_unused(m);
                alpaka::ignore_unused(n);
                alpaka::ignore_unused(k);
                alpaka::ignore_unused(alpha);
                alpaka::ignore_unused(a);
                alpaka::ignore_unused(lda);
                alpaka::ignore_unused(b);
                alpaka::ignore_unused(ldb);
                alpaka::ignore_unused(beta);
                alpaka::ignore_unused(c);
                alpaka::ignore_unused(ldc);

                // Reserve the buffer for the two blocks of A and B.
                return static_cast<std::size_t>(2u * block_thread_extent.prod() * thread_elem_extent.prod())
                    * sizeof(TElem);
            }
        };
    } // namespace traits
} // namespace alpaka

using TestAccs = alpaka::test::EnabledAccs<alpaka::DimInt<2u>, std::uint32_t>;

TEMPLATE_LIST_TEST_CASE("matMul", "[matMul]", TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    Idx const m(64u);
    Idx const n(79u);
    Idx const k(23u);

    using Val = std::uint32_t;
    using Vec2 = alpaka::Vec<Dim, Idx>;
    using DevAcc = alpaka::Dev<Acc>;
    using PltfAcc = alpaka::Pltf<DevAcc>;
    using QueueAcc = alpaka::test::DefaultQueue<alpaka::Dev<Acc>>;
    using PltfHost = alpaka::PltfCpu;
    using DevHost = alpaka::Dev<PltfHost>;
    using QueueHost = alpaka::QueueCpuNonBlocking;

    // Create the kernel function object.
    MatMulKernel kernel;

    // Get the host device.
    DevHost const dev_host = alpaka::getDevByIdx<PltfHost>(0u);

    // Get a queue on the host device.
    QueueHost queue_host(dev_host);

    // Select a device to execute on.
    DevAcc const dev_acc = alpaka::getDevByIdx<PltfAcc>(0u);

    // Get a queue on the accelerator device.
    QueueAcc queue_acc(dev_acc);

    // Specify the input matrix extents.
    Vec2 const extent_a(static_cast<Idx>(m), static_cast<Idx>(k));

    Vec2 const extent_b(static_cast<Idx>(k), static_cast<Idx>(n));

    // Result matrix is MxN. We create one worker per result matrix cell.
    Vec2 const extent_c(static_cast<Idx>(m), static_cast<Idx>(n));

    // Let alpaka calculate good block and grid sizes given our full problem extent.
    alpaka::WorkDivMembers<Dim, Idx> const work_div(alpaka::getValidWorkDiv<Acc>(
        dev_acc,
        extent_c,
        alpaka::Vec<Dim, Idx>::ones(),
        false,
        alpaka::GridBlockExtentSubDivRestrictions::EqualExtent));

    std::cout << "MatMulKernel("
              << "m:" << m << ", n:" << n << ", k:" << k << ", accelerator: " << alpaka::getAccName<Acc>()
              << ", kernel: " << typeid(kernel).name() << ", workDiv: " << work_div << ")" << std::endl;

    // Allocate the A and B matrices as std::vectors because this allows them to be filled with uint32_t(1).
    // alpaka::set only supports setting all bytes leading to a value of 16843009 in all elements.
    std::vector<Val> buf_a_host1d(m * k, static_cast<Val>(1));
    std::vector<Val> buf_b_host1d(k * n, static_cast<Val>(1));
    // Wrap the std::vectors into a memory buffer object.
    // For 1D data this would not be required because alpaka::copy is specialized for std::vector and std::array.
    // For multi dimensional data you could directly create them using alpaka::malloc<Type>(devHost, extent), which is
    // not used here. Instead we create a View to wrap the data.
    auto buf_a_host = alpaka::createView(dev_host, buf_a_host1d.data(), extent_a);
    auto buf_b_host = alpaka::createView(dev_host, buf_b_host1d.data(), extent_b);

    // Allocate C and set it to zero.
    auto buf_c_host = alpaka::allocBuf<Val, Idx>(dev_host, extent_c);
    alpaka::memset(queue_host, buf_c_host, 0u, extent_c);

    // Allocate the buffers on the accelerator.
    auto buf_a_acc = alpaka::allocBuf<Val, Idx>(dev_acc, extent_a);
    auto buf_b_acc = alpaka::allocBuf<Val, Idx>(dev_acc, extent_b);
    auto buf_c_acc = alpaka::allocBuf<Val, Idx>(dev_acc, extent_c);

    // Copy Host -> Acc.
    alpaka::memcpy(queue_acc, buf_a_acc, buf_a_host, extent_a);
    alpaka::memcpy(queue_acc, buf_b_acc, buf_b_host, extent_b);
    alpaka::wait(queue_host);
    alpaka::memcpy(queue_acc, buf_c_acc, buf_c_host, extent_c);

    // Create the kernel execution task.
    auto const task_kernel = alpaka::createTaskKernel<Acc>(
        work_div,
        kernel,
        m,
        n,
        k,
        static_cast<Val>(1),
        alpaka::getPtrNative(buf_a_acc),
        static_cast<Idx>(alpaka::getPitchBytes<1u>(buf_a_acc) / sizeof(Val)),
        alpaka::getPtrNative(buf_b_acc),
        static_cast<Idx>(alpaka::getPitchBytes<1u>(buf_b_acc) / sizeof(Val)),
        static_cast<Val>(1),
        alpaka::getPtrNative(buf_c_acc),
        static_cast<Idx>(alpaka::getPitchBytes<1u>(buf_c_acc) / sizeof(Val)));

    // Profile the kernel execution.
    std::cout << "Execution time: " << alpaka::test::integ::measureTaskRunTimeMs(queue_acc, task_kernel) << " ms"
              << std::endl;

    // Copy back the result.
    alpaka::memcpy(queue_acc, buf_c_host, buf_c_acc, extent_c);

    // Wait for the queue to finish the memory operation.
    alpaka::wait(queue_acc);

    // Assert that the results are correct.
    // When multiplying square matrices filled with ones, the result of each cell is the size of the matrix.
    auto const correct_result = static_cast<Val>(k);

    bool result_correct = true;
    auto const p_host_data = alpaka::getPtrNative(buf_c_host);
    for(Idx i(0u); i < m * n; ++i)
    {
        auto const& val(p_host_data[i]);
        if(val != correct_result)
        {
            std::cerr << "C[" << i << "] == " << val << " != " << correct_result << std::endl;
            result_correct = false;
        }
    }

    REQUIRE(result_correct);
}
