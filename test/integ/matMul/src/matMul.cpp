/* Copyright 2019-2021 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera, Bernhard Manfred Gruber
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
    //! \param A The accessor to the matrix A data.
    //! \param B The accessor to the matrix B data.
    //! \param C The accessor to the matrix C data.
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc,
        typename TMemoryHandleA,
        typename TMemoryHandleB,
        typename TMemoryHandleC,
        typename TElem,
        typename TIndex>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        TIndex const& m,
        TIndex const& n,
        TIndex const& k,
        TElem const& alpha,
        alpaka::experimental::Accessor<TMemoryHandleA, TElem, TIndex, 2, alpaka::experimental::ReadAccess> const A,
        alpaka::experimental::Accessor<TMemoryHandleB, TElem, TIndex, 2, alpaka::experimental::ReadAccess> const B,
        TElem const& beta,
        alpaka::experimental::Accessor<TMemoryHandleC, TElem, TIndex, 2, alpaka::experimental::ReadWriteAccess> const
            C) const -> void
    {
        static_assert(
            alpaka::Dim<TAcc>::value == 2u,
            "The accelerator used for the GemmAlpakaKernel has to be 2 dimensional!");

        // Column and row of C to calculate.
        auto const gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const& gridThreadIdxX = gridThreadIdx[1u];
        auto const& gridThreadIdxY = gridThreadIdx[0u];

        // Column and row inside the block of C to calculate.
        auto const blockThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
        auto const& blockThreadIdxX = blockThreadIdx[1u];
        auto const& blockThreadIdxY = blockThreadIdx[0u];

        // The block threads extent.
        auto const blockThreadExtent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        auto const& blockThreadExtentX = blockThreadExtent[1u];
        auto const& blockThreadExtentY = blockThreadExtent[0u];
        // ALPAKA_ASSERT(blockThreadExtentX == blockThreadExtentY);
        auto const& blockThreadExtentVal = blockThreadExtentX;

        // Shared memory used to store the current blocks of A and B.
        auto* const pBlockSharedA = alpaka::getDynSharedMem<TElem>(acc);
        auto* const pBlockSharedB = pBlockSharedA + blockThreadExtentX * blockThreadExtentY;

        auto const sharedBlockIdx1d = blockThreadIdxY * blockThreadExtentX + blockThreadIdxX;

        // If the element corresponding to the current thread is outside of the respective matrix.
        bool const insideA(gridThreadIdxY < m);
        bool const insideB(gridThreadIdxX < n);
        bool const insideC(insideA && insideB);

        TElem dotProduct(0);

        // Loop over all blocks of A and B that are required to compute the C block.
        auto const blockMulCount
            = static_cast<TIndex>(std::ceil(static_cast<float>(k) / static_cast<float>(blockThreadExtentVal)));
        for(TIndex k2(0u); k2 < blockMulCount; ++k2)
        {
            // Copy the current blocks of A and B into shared memory in parallel.
            // If the element of the current thread is outside of the matrix, zero is written into the shared memory.
            // This is possible because zero is a result neutral extension of the matrices regarding the dot product.
            auto const AIdxX = k2 * blockThreadExtentX + blockThreadIdxX;
            pBlockSharedA[sharedBlockIdx1d]
                = (((!insideA) || (AIdxX >= k)) ? static_cast<TElem>(0) : A(gridThreadIdxY, AIdxX));

            auto const BIdxY = k2 * blockThreadExtentY + blockThreadIdxY;
            pBlockSharedB[sharedBlockIdx1d]
                = (((!insideB) || (BIdxY >= k)) ? static_cast<TElem>(0) : B(BIdxY, gridThreadIdxX));

            // Synchronize to make sure the complete blocks are loaded before starting the computation.
            alpaka::syncBlockThreads(acc);

            // Not really necessary because we wrote zeros into those cells.
            // if(insideC)
            //{
            // Compute the dot products within shared memory.
            for(TIndex k3(0); k3 < blockThreadExtentVal; ++k3)
            {
                dotProduct += pBlockSharedA[blockThreadIdxY * blockThreadExtentX + k3]
                    * pBlockSharedB[k3 * blockThreadExtentY + blockThreadIdxX];
            }
            //}

            // Synchronize to make sure that the preceding computation is done before loading the next blocks of A and
            // B.
            alpaka::syncBlockThreads(acc);
        }

        // If the element is outside of the matrix it was only a helper thread that did not calculate any meaningful
        // results.
        if(insideC)
        {
            auto& c = C(gridThreadIdxY, gridThreadIdxX);
            c = alpha * dotProduct + beta * c;
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
            template<
                typename TVec,
                typename TIndex,
                typename TElem,
                typename TMemoryHandleA,
                typename TMemoryHandleB,
                typename TMemoryHandleC>
            ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
                MatMulKernel const& matMulKernel,
                TVec const& blockThreadExtent,
                TVec const& threadElemExtent,
                TIndex const& m,
                TIndex const& n,
                TIndex const& k,
                TElem const& alpha,
                alpaka::experimental::
                    Accessor<TMemoryHandleA, TElem, TIndex, 2, alpaka::experimental::ReadAccess> const A,
                alpaka::experimental::
                    Accessor<TMemoryHandleB, TElem, TIndex, 2, alpaka::experimental::ReadAccess> const B,
                TElem const& beta,
                alpaka::experimental::
                    Accessor<TMemoryHandleC, TElem, TIndex, 2, alpaka::experimental::ReadWriteAccess> const C)
            {
                alpaka::ignore_unused(matMulKernel);
                alpaka::ignore_unused(m);
                alpaka::ignore_unused(n);
                alpaka::ignore_unused(k);
                alpaka::ignore_unused(alpha);
                alpaka::ignore_unused(A);
                alpaka::ignore_unused(B);
                alpaka::ignore_unused(beta);
                alpaka::ignore_unused(C);

                // Reserve the buffer for the two blocks of A and B.
                return static_cast<std::size_t>(2u * blockThreadExtent.prod() * threadElemExtent.prod())
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
    DevHost const devHost = alpaka::getDevByIdx<PltfHost>(0u);

    // Get a queue on the host device.
    QueueHost queueHost(devHost);

    // Select a device to execute on.
    DevAcc const devAcc = alpaka::getDevByIdx<PltfAcc>(0u);

    // Get a queue on the accelerator device.
    QueueAcc queueAcc(devAcc);

    // Specify the input matrix extents.
    Vec2 const extentA(static_cast<Idx>(m), static_cast<Idx>(k));

    Vec2 const extentB(static_cast<Idx>(k), static_cast<Idx>(n));

    // Result matrix is MxN. We create one worker per result matrix cell.
    Vec2 const extentC(static_cast<Idx>(m), static_cast<Idx>(n));

    // Let alpaka calculate good block and grid sizes given our full problem extent.
    alpaka::WorkDivMembers<Dim, Idx> const workDiv(alpaka::getValidWorkDiv<Acc>(
        devAcc,
        extentC,
        alpaka::Vec<Dim, Idx>::ones(),
        false,
        alpaka::GridBlockExtentSubDivRestrictions::EqualExtent));

    std::cout << "MatMulKernel("
              << "m:" << m << ", n:" << n << ", k:" << k << ", accelerator: " << alpaka::getAccName<Acc>()
              << ", kernel: " << typeid(kernel).name() << ", workDiv: " << workDiv << ")" << std::endl;

    // Allocate the A and B matrices as std::vectors because this allows them to be filled with uint32_t(1).
    // alpaka::set only supports setting all bytes leading to a value of 16843009 in all elements.
    std::vector<Val> bufAHost1d(m * k, static_cast<Val>(1));
    std::vector<Val> bufBHost1d(k * n, static_cast<Val>(1));
    // Wrap the std::vectors into a memory buffer object.
    // For 1D data this would not be required because alpaka::copy is specialized for std::vector and std::array.
    // For multi dimensional data you could directly create them using alpaka::malloc<Type>(devHost, extent), which is
    // not used here. Instead we create a View to wrap the data.
    auto bufAHost = alpaka::createView(devHost, bufAHost1d.data(), extentA);
    auto bufBHost = alpaka::createView(devHost, bufBHost1d.data(), extentB);

    // Allocate C and set it to zero.
    auto bufCHost = alpaka::allocBuf<Val, Idx>(devHost, extentC);
    alpaka::memset(queueHost, bufCHost, 0u, extentC);

    // Allocate the buffers on the accelerator.
    auto bufAAcc = alpaka::allocBuf<Val, Idx>(devAcc, extentA);
    auto bufBAcc = alpaka::allocBuf<Val, Idx>(devAcc, extentB);
    auto bufCAcc = alpaka::allocBuf<Val, Idx>(devAcc, extentC);

    // Copy Host -> Acc.
    alpaka::memcpy(queueAcc, bufAAcc, bufAHost, extentA);
    alpaka::memcpy(queueAcc, bufBAcc, bufBHost, extentB);
    alpaka::wait(queueHost);
    alpaka::memcpy(queueAcc, bufCAcc, bufCHost, extentC);

    // Create the kernel execution task.
    auto const taskKernel = alpaka::createTaskKernel<Acc>(
        workDiv,
        kernel,
        m,
        n,
        k,
        static_cast<Val>(1),
        alpaka::experimental::readAccess(bufAAcc),
        alpaka::experimental::readAccess(bufBAcc),
        static_cast<Val>(1),
        alpaka::experimental::access(bufCAcc));

    // Profile the kernel execution.
    std::cout << "Execution time: " << alpaka::test::integ::measureTaskRunTimeMs(queueAcc, taskKernel) << " ms"
              << std::endl;

    // Copy back the result.
    alpaka::memcpy(queueAcc, bufCHost, bufCAcc, extentC);

    // Wait for the queue to finish the memory operation.
    alpaka::wait(queueAcc);

    // Assert that the results are correct.
    // When multiplying square matrices filled with ones, the result of each cell is the size of the matrix.
    auto const correctResult = static_cast<Val>(k);

    bool resultCorrect = true;
    auto const hostData = alpaka::experimental::readAccess(bufCHost);
    for(Idx row = 0u; row < m; ++row)
    {
        for(Idx col(0u); col < n; ++col)
        {
            auto const& val(hostData(row, col));
            if(val != correctResult)
            {
                std::cerr << "C[" << row << "," << col << "] == " << val << " != " << correctResult << std::endl;
                resultCorrect = false;
            }
        }
    }

    REQUIRE(resultCorrect);
}
