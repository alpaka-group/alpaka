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

#include <alpaka/alpaka.hpp>                // alpaka::exec::create

#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
    #if BOOST_COMP_MSVC
        #pragma warning(push)
        #pragma warning(disable: 4512)      // boost/program_options/options_description.hpp(265): warning C4512: 'boost::program_options::options_description': assignment operator was implicitly defined as deleted
    #endif

    #include <boost/program_options.hpp>    // boost::program_options

    #if BOOST_COMP_MSVC
        #pragma warning(pop)
    #endif
#endif

#include <boost/core/ignore_unused.hpp>     // boost::ignore_unused

#include <chrono>                           // std::chrono::high_resolution_clock
#include <cassert>                          // assert
#include <iostream>                         // std::cout
#include <vector>                           // std::vector
#include <typeinfo>                         // typeid
#include <utility>                          // std::forward
#include <functional>                       // std::placeholders
#include <unordered_map>                    // std::unordered_map

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
    template<
        typename TAcc,
        typename TElem,
        typename TIndex>
    ALPAKA_FCT_ACC auto operator()(
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
        /*static_assert(alpaka::dim::DimT<TAcc>::value == 2u,
            "The accelerator used for with MatMulKernel has to be 2 dimensional!");*/

        // Column and row of C to calculate.
        auto const v2uiGridThreadIdx(alpaka::subVecEnd<alpaka::dim::Dim2>(acc.template getIdx<alpaka::Grid, alpaka::Threads>()));
        auto const & uiGridThreadIdxX(v2uiGridThreadIdx[1u]);
        auto const & uiGridThreadIdxY(v2uiGridThreadIdx[0u]);

        // Column and row inside the block of C to calculate.
        auto const v2uiBlockThreadIdx(alpaka::subVecEnd<alpaka::dim::Dim2>(acc.template getIdx<alpaka::Block, alpaka::Threads>()));
        auto const & uiBlockThreadIdxX(v2uiBlockThreadIdx[1u]);
        auto const & uiBlockThreadIdxY(v2uiBlockThreadIdx[0u]);

        // The block threads extents.
        auto const v2uiBlockThreadsExtents(alpaka::subVecEnd<alpaka::dim::Dim2>(acc.template getWorkDiv<alpaka::Block, alpaka::Threads>()));
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
            C[uiIdxC] =  alpha * fCSum + beta * C[uiIdxC];
        }
    }
};

namespace alpaka
{
    namespace traits
    {
        namespace kernel
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
                    typename TIndex,
                    typename TElem>
                ALPAKA_FCT_HOST static auto getBlockSharedExternMemSizeBytes(
                    alpaka::Vec3<> const & v3uiBlockThreadsExtents,
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
                -> UInt
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
                    return 2u * v3uiBlockThreadsExtents.prod() * sizeof(TElem);
                }
            };
        }
    }
}

//-----------------------------------------------------------------------------
//! Profiles the given kernel.
//-----------------------------------------------------------------------------
template<
    typename TExec,
    typename TKernelFunctor,
    typename... TArgs>
auto profileKernelExec(
    TExec const & exec,
    TKernelFunctor && kernelFunctor,
    TArgs && ... args)
-> void
{
    std::cout
        << "profileKernelExec("
        << " kernelExecutor: " << typeid(TExec).name()
        << ")" << std::endl;

    auto const tpStart(std::chrono::high_resolution_clock::now());

    // Execute the kernel functor.
    exec(std::forward<TKernelFunctor>(kernelFunctor), std::forward<TArgs>(args)...);

    // Wait for the stream to finish the kernel execution to measure its run time.
    alpaka::wait::wait(alpaka::stream::getStream(exec));

    auto const tpEnd(std::chrono::high_resolution_clock::now());

    auto const durElapsed(tpEnd - tpStart);

    std::cout << "Execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(durElapsed).count() << " ms" << std::endl;
}

//#############################################################################
//! Profiles the example kernel and checks the result.
//#############################################################################
struct MatMulTester
{
    template<
        typename TAcc>
    auto operator()(
        std::size_t const & m,
        std::size_t const & n,
        std::size_t const & k,
        bool const & bAdaptiveBlockThreadExtent)
    -> void
    {
        std::cout << std::endl;
        std::cout << "################################################################################" << std::endl;

        // Create the kernel functor.
        MatMulKernel kernel;

        // Get the host device.
        auto devHost(alpaka::devs::cpu::getDev());

        // Select a device to execute on.
        alpaka::dev::DevT<TAcc> devAcc(
            alpaka::dev::DevManT<TAcc>::getDevByIdx(0));

        // Get a stream on this device.
        alpaka::stream::StreamT<TAcc> stream(
            alpaka::stream::create(devAcc));

        alpaka::Vec2<> const v2uiExtentsA(
            static_cast<alpaka::Vec2<>::Val>(m),
            static_cast<alpaka::Vec2<>::Val>(k));

        alpaka::Vec2<> const v2uiExtentsB(
            static_cast<alpaka::Vec2<>::Val>(k),
            static_cast<alpaka::Vec2<>::Val>(n));

        // Result matrix is MxN. We create one worker per result matrix cell.
        alpaka::Vec2<> const v2uiExtentsC(
            static_cast<alpaka::Vec2<>::Val>(m),
            static_cast<alpaka::Vec2<>::Val>(n));

        // Let alpaka calculate good block and grid sizes given our full problem extents.
        alpaka::workdiv::BasicWorkDiv workDiv(
            bAdaptiveBlockThreadExtent
            ? alpaka::workdiv::getValidWorkDiv<boost::mpl::vector<TAcc>>(v2uiExtentsC, false)
            : alpaka::workdiv::getValidWorkDiv<alpaka::accs::EnabledAccs>(v2uiExtentsC, false));
        // Assure that the extents are square.
        auto const uiMinExtent(std::min(workDiv.m_v3uiBlockThreadExtents[1u], workDiv.m_v3uiBlockThreadExtents[2u]));
        workDiv.m_v3uiGridBlockExtents[1u] = static_cast<alpaka::Vec3<>::Val>(std::ceil(static_cast<double>(m) / static_cast<double>(uiMinExtent)));
        workDiv.m_v3uiBlockThreadExtents[1u] = uiMinExtent;
        workDiv.m_v3uiGridBlockExtents[2u] = static_cast<alpaka::Vec3<>::Val>(std::ceil(static_cast<double>(n) / static_cast<double>(uiMinExtent)));
        workDiv.m_v3uiBlockThreadExtents[2u] = uiMinExtent;

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
        // alpaka::mem::set only supports setting all bytes leading to a value of 16843009 in all elements.
        std::vector<std::uint32_t> vuiA(m * k, 1u);
        std::vector<std::uint32_t> vuiB(k * n, 1u);
        // Wrap the std::vectors into a memory buffer object.
        // For 1D data this would not be required because alpaka::mem::copy is specialized for std::vector and std::array.
        // For multi dimensional data you could directly create them using alpaka::mem::alloc<Type>(devHost, extents), which is not used here.
        // Instead we use BufPlainPtrWrapper to wrap the data.
        using bufWrapper = alpaka::mem::BufPlainPtrWrapper<
            std::uint32_t,
            alpaka::dim::Dim2,
            std::decay<decltype(devHost)>::type>;
        bufWrapper bufAHost(vuiA.data(), devHost, v2uiExtentsA);
        bufWrapper bufBHost(vuiB.data(), devHost, v2uiExtentsB);

        // Allocate C and set it to zero.
        auto bufCHost(alpaka::mem::alloc<std::uint32_t>(devHost, v2uiExtentsC));
        alpaka::mem::set(bufCHost, 0u, v2uiExtentsC);

        // Allocate the buffers on the accelerator.
        auto bufAAcc(alpaka::mem::alloc<std::uint32_t>(devAcc, v2uiExtentsA));
        auto bufBAcc(alpaka::mem::alloc<std::uint32_t>(devAcc, v2uiExtentsB));
        auto bufCAcc(alpaka::mem::alloc<std::uint32_t>(devAcc, v2uiExtentsC));

        // Copy Host -> Acc.
        alpaka::mem::copy(bufAAcc, bufAHost, v2uiExtentsA, stream);
        alpaka::mem::copy(bufBAcc, bufBHost, v2uiExtentsB, stream);
        alpaka::mem::copy(bufCAcc, bufCHost, v2uiExtentsC, stream);

        // Create the executor.
        auto exec(alpaka::exec::create<TAcc>(workDiv, stream));
        // Profile the kernel execution.
        profileKernelExec(
            exec,
            kernel,
            static_cast<std::uint32_t>(m),
            static_cast<std::uint32_t>(n),
            static_cast<std::uint32_t>(k),
            static_cast<std::uint32_t>(1u),
            alpaka::mem::getPtrNative(bufAAcc),
            alpaka::mem::getPitchElements<1u, std::uint32_t>(bufAAcc),
            alpaka::mem::getPtrNative(bufBAcc),
            alpaka::mem::getPitchElements<1u, std::uint32_t>(bufBAcc),
            static_cast<std::uint32_t>(1u),
            alpaka::mem::getPtrNative(bufCAcc),
            alpaka::mem::getPitchElements<1u, std::uint32_t>(bufCAcc));

        // Copy back the result.
        alpaka::mem::copy(bufCHost, bufCAcc, v2uiExtentsC, stream);

        // Wait for the stream to finish the memory operation.
        alpaka::wait::wait(stream);

        // Assert that the results are correct.
        // When multiplying square matrices filled with ones, the result of each cell is the size of the matrix.
        std::uint32_t const uiCorrectResult(static_cast<std::uint32_t>(k));

        bool bResultCorrect(true);
        auto const pHostData(alpaka::mem::getPtrNative(bufCHost));
        for(std::size_t i(0u);
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
auto main(
    int argc,
    char * argv[])
-> int
{
    try
    {
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
        // Declare the supported options.
        boost::program_options::options_description desc("Available options");
        desc.add_options()
            ("help",
            "Prints the help message.")
            ("adaptiveBlockThreadExtent,a",
            boost::program_options::value<bool>()->default_value(true),
            "If the size of a block is the minimum of all enabled accelerators (false), or adaptive to the current accelerator (true).")
        ;
        boost::program_options::variables_map vm;
        boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
        boost::program_options::notify(vm);

        if(vm.count("help") > 0u)
        {
            std::cout << desc << std::endl;
            return EXIT_FAILURE;
        }
        else
#endif
        {
            std::cout << std::endl;
            std::cout << "################################################################################" << std::endl;
            std::cout << "                              alpaka matMul test                                " << std::endl;
            std::cout << "################################################################################" << std::endl;
            std::cout << std::endl;

            // Logs the enabled accelerators.
            alpaka::accs::writeEnabledAccs(std::cout);

            std::cout << std::endl;

#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
            assert(vm.count("adaptiveBlockThreadExtent") > 0);
            bool const bAdaptiveBlockThreadExtent(vm["adaptiveBlockThreadExtent"].as<bool>());
#else
            bool const bAdaptiveBlockThreadExtent(true);
#endif
            std::cout << "Adaptive block thread size:" << bAdaptiveBlockThreadExtent << std::endl;

            MatMulTester matMulTester;

            // For different matrix sizes.
#if ALPAKA_INTEGRATION_TEST
            for(std::size_t m(1u); m <= 64u; m *= 8u)
            {
                for(std::size_t n(1u); n <= 79u; n *= 79u)
                {
                    for(std::size_t k(1u); k <= 23u; k *= 23u)
                    {
#else
            for(std::size_t m(1u); m <= 1024u; m *= 4u)
            {
                for(std::size_t n(1u); n <= 1024u; n *= 4u)
                {
                    for(std::size_t k(1u); k <= 1024u; k *= 4u)
                    {
#endif
                        std::cout << std::endl;

                        // Execute the kernel on all enabled accelerators.
                        alpaka::forEachType<alpaka::accs::EnabledAccs>(
                            matMulTester,
                            m, n, k,
                            bAdaptiveBlockThreadExtent);
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
