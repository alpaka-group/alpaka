/**
 * \file
 * Copyright 2014-2015 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * alpaka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * alpaka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with alpaka.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <alpaka/alpaka.hpp>                        // alpaka::exec::create
#include <alpaka/examples/MeasureKernelRunTime.hpp> // measureKernelRunTimeMs
#include <alpaka/examples/accs/EnabledAccs.hpp>     // EnabledAccs

#include <boost/core/ignore_unused.hpp>             // boost::ignore_unused

#include <chrono>                                   // std::chrono::high_resolution_clock
#include <cassert>                                  // assert
#include <iostream>                                 // std::cout
#include <vector>                                   // std::vector
#include <typeinfo>                                 // typeid
#include <utility>                                  // std::forward
#include <functional>                               // std::placeholders
#include <unordered_map>                            // std::unordered_map

//#############################################################################
//! A matrix multiplication kernel.
//! Computes C += A*B. LxM * MxN -> LxN
//! This is an adaption of the algorithm from the CUDA developers guide.
//! \tparam TAcc The accelerator environment to be executed on.
//#############################################################################
class MatMulKernel
{
public:
    //-----------------------------------------------------------------------------
    //! The kernel entrypoint.
    //!
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
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc,
        typename TElem,
        typename TIndex>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        TIndex const & m,
        TIndex const & n,
        TIndex const & k,
        TElem const & alpha,
        TElem const * const A,
        TIndex const & lda,
        TElem const * const B,
        TIndex const & ldb,
        TElem const & beta,
        TElem * const C,
        TIndex const & ldc) const
    -> void
    {
        static_assert(alpaka::dim::Dim<TAcc>::value == 2u,
            "The accelerator used for with MatMulKernel has to be 2 dimensional!");

        // Column and row of C to calculate.
        auto const v2uiGridThreadIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc));
        auto const & uiGridThreadIdxX(v2uiGridThreadIdx[1u]);
        auto const & uiGridThreadIdxY(v2uiGridThreadIdx[0u]);

        // Column and row inside the block of C to calculate.
        auto const v2uiBlockThreadIdx(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc));
        auto const & uiBlockThreadIdxX(v2uiBlockThreadIdx[1u]);
        auto const & uiBlockThreadIdxY(v2uiBlockThreadIdx[0u]);

        // The block threads extents.
        auto const v2uiBlockThreadsExtents(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(acc));
        auto const & uiBlockThreadsExtentX(v2uiBlockThreadsExtents[1u]);
        auto const & uiBlockThreadsExtentY(v2uiBlockThreadsExtents[0u]);
        //assert(uiBlockThreadsExtentX == uiBlockThreadsExtentY);
        auto const & uiBlockThreadsExtent(uiBlockThreadsExtentX);

        // Shared memory used to store the current blocks of A and B.
        auto * const pBlockSharedA(acc.template getBlockSharedExternMem<TElem>());
        auto * const pBlockSharedB(pBlockSharedA + uiBlockThreadsExtentX*uiBlockThreadsExtentY);

        auto const uiSharedBlockIdx1d(uiBlockThreadIdxY*uiBlockThreadsExtentX + uiBlockThreadIdxX);

        bool const bInsideA(uiGridThreadIdxY < m);
        bool const bInsideB(uiGridThreadIdxX < n);
        bool const bInsideC(bInsideA && bInsideB);

        TElem fCSum(0);

        // Loop over all blocks of A and B that are required to compute the C block.
        auto const uiBlockMulCount(static_cast<TIndex>(std::ceil(static_cast<float>(k)/static_cast<float>(uiBlockThreadsExtent))));
        for(TIndex k2(0u); k2 < uiBlockMulCount; ++k2)
        {
            // Copy data to shared memory.
            auto const uiAIdxX(k2*uiBlockThreadsExtentX + uiBlockThreadIdxX);
            auto const uiAIdx1d(uiGridThreadIdxY*lda + uiAIdxX);
            pBlockSharedA[uiSharedBlockIdx1d] = (
                ((!bInsideA) || (uiAIdxX>=k))
                ? static_cast<TElem>(0)
                : A[uiAIdx1d]);

            auto const uiBIdxY(k2*uiBlockThreadsExtentY + uiBlockThreadIdxY);
            auto const uiBIdx1d(uiBIdxY*ldb + uiGridThreadIdxX);
            pBlockSharedB[uiSharedBlockIdx1d] = (
                ((!bInsideB) || (uiBIdxY>=k))
                ? static_cast<TElem>(0)
                : B[uiBIdx1d]);

            // Synchronize to make sure the sub-matrices are loaded before starting the computation.
            acc.syncBlockThreads();

            // Not really necessary because we wrote zeros into those cells.
            //if(bInsideC)
            //{
                // Dyadic product within shared memory.
                for(TIndex k3(0); k3 < uiBlockThreadsExtent; ++k3)
                {
                    fCSum += pBlockSharedA[uiBlockThreadIdxY*uiBlockThreadsExtentX + k3]
                        * pBlockSharedB[k3*uiBlockThreadsExtentY + uiBlockThreadIdxX];
                }
            //}

            // Synchronize to make sure that the preceding computation is done before loading two new sub-matrices of A and B in the next iteration.
            acc.syncBlockThreads();
        }

        // If the element is outside of the matrix it was only a helper thread that did not calculate any meaningful results
        if(bInsideC)
        {
            auto const uiIdxC(uiGridThreadIdxY*ldc + uiGridThreadIdxX);
            C[uiIdxC] = alpha * fCSum + beta * C[uiIdxC];
        }
    }
};

namespace alpaka
{
    namespace kernel
    {
        namespace traits
        {
            //#############################################################################
            //! The trait for getting the size of the block shared extern memory for a kernel.
            //#############################################################################
            template<
                typename TAcc>
            struct BlockSharedExternMemSizeBytes<
                MatMulKernel,
                TAcc>
            {
                //-----------------------------------------------------------------------------
                //! \return The size of the shared memory allocated for a block.
                //-----------------------------------------------------------------------------
                template<
                    typename TVec,
                    typename TIndex,
                    typename TElem>
                ALPAKA_FN_HOST static auto getBlockSharedExternMemSizeBytes(
                    TVec const & vuiBlockThreadsExtents,
                    TIndex const & m,
                    TIndex const & n,
                    TIndex const & k,
                    TElem const & alpha,
                    TElem const * const A,
                    TIndex const & lda,
                    TElem const * const B,
                    TIndex const & ldb,
                    TElem const & beta,
                    TElem * const C,
                    TIndex const & ldc)
                -> TIndex
                {
                    boost::ignore_unused(m);
                    boost::ignore_unused(n);
                    boost::ignore_unused(k);
                    boost::ignore_unused(alpha);
                    boost::ignore_unused(A);
                    boost::ignore_unused(lda);
                    boost::ignore_unused(B);
                    boost::ignore_unused(ldb);
                    boost::ignore_unused(beta);
                    boost::ignore_unused(C);
                    boost::ignore_unused(ldc);

                    // Reserve the buffer for the two blocks of A and B.
                    return 2u * vuiBlockThreadsExtents.prod() * sizeof(TElem);
                }
            };
        }
    }
}

//#############################################################################
//! Profiles the example kernel and checks the result.
//#############################################################################
struct MatMulTester
{
    template<
        typename TAcc,
        typename TSize>
    auto operator()(
        TSize const & m,
        TSize const & n,
        TSize const & k)
    -> void
    {
        std::cout << std::endl;
        std::cout << "################################################################################" << std::endl;

        using Val = std::uint32_t;

        // Create the kernel function object.
        MatMulKernel kernel;

        // Get the host device.
        auto devHost(alpaka::dev::cpu::getDev());

        // Select a device to execute on.
        alpaka::dev::Dev<TAcc> devAcc(
            alpaka::dev::DevMan<TAcc>::getDevByIdx(0));

        // Get a stream on this device.
        alpaka::stream::Stream<alpaka::dev::Dev<TAcc>> stream(
            alpaka::stream::create(devAcc));

        alpaka::Vec2<TSize> const v2uiExtentsA(
            static_cast<TSize>(m),
            static_cast<TSize>(k));

        alpaka::Vec2<TSize> const v2uiExtentsB(
            static_cast<TSize>(k),
            static_cast<TSize>(n));

        // Result matrix is MxN. We create one worker per result matrix cell.
        alpaka::Vec2<TSize> const v2uiExtentsC(
            static_cast<TSize>(m),
            static_cast<TSize>(n));

        // Let alpaka calculate good block and grid sizes given our full problem extents.
        alpaka::workdiv::WorkDivMembers<alpaka::dim::DimInt<2u>, TSize> const workDiv(
            alpaka::workdiv::getValidWorkDiv<TAcc>(
                devAcc,
                v2uiExtentsC,
                false,
                alpaka::workdiv::GridBlockExtentsSubDivRestrictions::EqualExtents));

        std::cout
            << "profileAcceleratedMatMulKernel("
            << "m:" << m
            << ", n:" << n
            << ", k:" << k
            << ", accelerator: " << alpaka::acc::getAccName<TAcc>()
            << ", kernel: " << typeid(kernel).name()
            << ", workDiv: " << workDiv
            << ")" << std::endl;

        // Allocate the A and B matrices as st::vectors because this allows them to be filled with uint32_t(1).
        // alpaka::mem::view::set only supports setting all bytes leading to a value of 16843009 in all elements.
        std::vector<Val> vuiA(m * k, static_cast<Val>(1));
        std::vector<Val> vuiB(k * n, static_cast<Val>(1));
        // Wrap the std::vectors into a memory buffer object.
        // For 1D data this would not be required because alpaka::mem::view::copy is specialized for std::vector and std::array.
        // For multi dimensional data you could directly create them using alpaka::mem::buf::alloc<Type>(devHost, extents), which is not used here.
        // Instead we use BufPlainPtrWrapper to wrap the data.
        using BufWrapper = alpaka::mem::buf::BufPlainPtrWrapper<
            std::decay<decltype(devHost)>::type,
            Val,
            alpaka::dim::DimInt<2u>,
            TSize>;
        BufWrapper bufAHost(vuiA.data(), devHost, v2uiExtentsA);
        BufWrapper bufBHost(vuiB.data(), devHost, v2uiExtentsB);

        // Allocate C and set it to zero.
        auto bufCHost(alpaka::mem::buf::alloc<Val, TSize>(devHost, v2uiExtentsC));
        alpaka::mem::view::set(bufCHost, 0u, v2uiExtentsC);

        // Allocate the buffers on the accelerator.
        auto bufAAcc(alpaka::mem::buf::alloc<Val, TSize>(devAcc, v2uiExtentsA));
        auto bufBAcc(alpaka::mem::buf::alloc<Val, TSize>(devAcc, v2uiExtentsB));
        auto bufCAcc(alpaka::mem::buf::alloc<Val, TSize>(devAcc, v2uiExtentsC));

        // Copy Host -> Acc.
        alpaka::mem::view::copy(bufAAcc, bufAHost, v2uiExtentsA, stream);
        alpaka::mem::view::copy(bufBAcc, bufBHost, v2uiExtentsB, stream);
        alpaka::mem::view::copy(bufCAcc, bufCHost, v2uiExtentsC, stream);

        // Create the executor.
        auto exec(alpaka::exec::create<TAcc>(workDiv, stream));
        // Profile the kernel execution.
        std::cout << "Execution time: "
            << alpaka::examples::measureKernelRunTimeMs(
                exec,
                kernel,
                m,
                n,
                k,
                static_cast<Val>(1),
                alpaka::mem::view::getPtrNative(bufAAcc),
                static_cast<TSize>(alpaka::mem::view::getPitchBytes<1u>(bufAAcc) / sizeof(Val)),
                alpaka::mem::view::getPtrNative(bufBAcc),
                static_cast<TSize>(alpaka::mem::view::getPitchBytes<1u>(bufBAcc) / sizeof(Val)),
                static_cast<Val>(1),
                alpaka::mem::view::getPtrNative(bufCAcc),
                static_cast<TSize>(alpaka::mem::view::getPitchBytes<1u>(bufCAcc) / sizeof(Val)))
            << " ms"
            << std::endl;

        // Copy back the result.
        alpaka::mem::view::copy(bufCHost, bufCAcc, v2uiExtentsC, stream);

        // Wait for the stream to finish the memory operation.
        alpaka::wait::wait(stream);

        // Assert that the results are correct.
        // When multiplying square matrices filled with ones, the result of each cell is the size of the matrix.
        auto const uiCorrectResult(static_cast<Val>(k));

        bool bResultCorrect(true);
        auto const pHostData(alpaka::mem::view::getPtrNative(bufCHost));
        for(TSize i(0u);
            i < m * n;
            ++i)
        {
            auto const & uiVal(pHostData[i]);
            if(uiVal != uiCorrectResult)
            {
                std::cout << "C[" << i << "] == " << uiVal << " != " << uiCorrectResult << std::endl;
                bResultCorrect = false;
            }
        }

        if(bResultCorrect)
        {
            std::cout << "Execution results correct!" << std::endl;
        }

        std::cout << "################################################################################" << std::endl;

        bAllResultsCorrect = bAllResultsCorrect && bResultCorrect;
    }

public:
    bool bAllResultsCorrect = true;
};

//-----------------------------------------------------------------------------
//! Program entry point.
//-----------------------------------------------------------------------------
auto main()
-> int
{
    try
    {
        {
            std::cout << std::endl;
            std::cout << "################################################################################" << std::endl;
            std::cout << "                              alpaka matMul test                                " << std::endl;
            std::cout << "################################################################################" << std::endl;
            std::cout << std::endl;

            // Logs the enabled accelerators.
            alpaka::examples::accs::writeEnabledAccs<alpaka::dim::DimInt<2u>, std::uint32_t>(std::cout);

            std::cout << std::endl;

            MatMulTester matMulTester;

            // For different matrix sizes.
#if ALPAKA_INTEGRATION_TEST
            for(std::uint32_t m(1u); m <= 64u; m *= 8u)
            {
                for(std::uint32_t n(1u); n <= 79u; n *= 79u)
                {
                    for(std::uint32_t k(1u); k <= 23u; k *= 23u)
                    {
#else
            for(std::uint32_t m(1u); m <= 1024u; m *= 4u)
            {
                for(std::uint32_t n(1u); n <= 1024u; n *= 4u)
                {
                    for(std::uint32_t k(1u); k <= 1024u; k *= 4u)
                    {
#endif
                        std::cout << std::endl;

                        // Execute the kernel on all enabled accelerators.
                        alpaka::forEachType<
                            alpaka::examples::accs::EnabledAccs<alpaka::dim::DimInt<2u>, std::uint32_t>>(
                                matMulTester,
                                m, n, k);
                    }
                }
            }
            return matMulTester.bAllResultsCorrect ? EXIT_SUCCESS : EXIT_FAILURE;
        }
    }
    catch(std::exception const & e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch(...)
    {
        std::cerr << "Unknown Exception" << std::endl;
        return EXIT_FAILURE;
    }
}
