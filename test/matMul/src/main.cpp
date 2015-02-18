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

#include <alpaka/alpaka.hpp>                // alpaka::createKernelExecutor<...>

#include <chrono>                           // std::chrono::high_resolution_clock
#include <cassert>                          // assert
#include <iostream>                         // std::cout
#include <vector>                           // std::vector
#include <typeinfo>                         // typeid
#include <utility>                          // std::forward
#include <functional>                       // std::placeholders
#include <unordered_map>                    // std::unordered_map

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

#include <boost/mpl/for_each.hpp>           // boost::mpl::for_each

//#############################################################################
//! A matrix multiplication kernel.
//! Computes C += A*B. LxM * MxN -> LxN
//! This is an adaption of the algorithm from the CUDA developers guide.
//! \tparam TAcc The accelerator environment to be executed on.
//#############################################################################
template<
    typename TAcc = alpaka::IAcc<>>
class MatMulKernel
{
public:
    //-----------------------------------------------------------------------------
    //! The kernel entrypoint.
    //!
    //! \tparam TElem The matrix element type.
    //! \param acc The accelerator to be executed on.
    //! \param uiL The height of the A matrix.
    //! \param uiM The width of the A and height of the B matrix.
    //! \param uiN The width of the B matrix.
    //! \param A The pointer to the matrix A data.
    //! \param uiPitchElemA The pitch of the A matrix in elements.
    //! \param B The pointer to the matrix B data.
    //! \param uiPitchElemB The pitch of the B matrix in elements.
    //! \param C The pointer to the matrix C data.
    //! \param uiPitchElemC The pitch of the C matrix in elements.
    //-----------------------------------------------------------------------------
    template<
        typename TElem,
        typename TIndex>
    ALPAKA_FCT_ACC void operator()(
        TAcc const & acc,
        TIndex const & uiL,
        TIndex const & uiM,
        TIndex const & uiN,
        TElem const * const A,
        TIndex const & uiPitchElemA,
        TElem const * const B,
        TIndex const & uiPitchElemB,
        TElem * const C,
        TIndex const & uiPitchElemC) const
    {
        // Column and row of C to calculate.
        auto const v2uiGridKernelIdx(acc.template getIdx<alpaka::Grid, alpaka::Kernels>().template subvec<2u>());
        auto const & uiGridKernelIdxX(v2uiGridKernelIdx[0u]);
        auto const & uiGridKernelIdxY(v2uiGridKernelIdx[1u]);

        // Column and row inside the block of C to calculate.
        auto const v2uiBlockKernelIdx(acc.template getIdx<alpaka::Block, alpaka::Kernels>().template subvec<2u>());
        auto const & uiBlockKernelIdxX(v2uiBlockKernelIdx[0u]);
        auto const & uiBlockKernelIdxY(v2uiBlockKernelIdx[1u]);

        // The block kernels extents.
        auto const v2uiBlockKernelsExtents(acc.template getWorkDiv<alpaka::Block, alpaka::Kernels>().template subvec<2u>());
        auto const & uiBlockKernelsExtentX(v2uiBlockKernelsExtents[0u]);
        auto const & uiBlockKernelsExtentY(v2uiBlockKernelsExtents[1u]);
        //assert(uiBlockKernelsExtentX == uiBlockKernelsExtentY);
        auto const & uiBlockKernelsExtent(uiBlockKernelsExtentX);

        // Shared memory used to store the current blocks of A and B.
        auto * const pBlockSharedA(acc.template getBlockSharedExternMem<TElem>());
        auto * const pBlockSharedB(pBlockSharedA + uiBlockKernelsExtentX*uiBlockKernelsExtentY);

        auto const uiSharedBlockIdx1d(uiBlockKernelIdxY*uiBlockKernelsExtentX + uiBlockKernelIdxX);
        
        bool const bInsideA(uiGridKernelIdxY < uiL);
        bool const bInsideB(uiGridKernelIdxX < uiN);
        bool const bInsideC(bInsideA && bInsideB);

        TElem fCSum(0);

        // Loop over all blocks of A and B that are required to compute the C block.
        auto const uiBlockMulCount(static_cast<TIndex>(std::ceil(static_cast<float>(uiM)/static_cast<float>(uiBlockKernelsExtent))));
        for(TIndex l(0u); l < uiBlockMulCount; ++l)
        {
            // Copy data to shared memory.
            auto const uiAIdxX(l*uiBlockKernelsExtentX + uiBlockKernelIdxX);
            auto const uiAIdx1d(uiGridKernelIdxY*uiPitchElemA + uiAIdxX);
            pBlockSharedA[uiSharedBlockIdx1d] = (
                ((!bInsideA) || (uiAIdxX>=uiM))
                ? static_cast<TElem>(0)
                : A[uiAIdx1d]);

            auto const uiBIdxY(l*uiBlockKernelsExtentY + uiBlockKernelIdxY);
            auto const uiBIdx1d(uiBIdxY*uiPitchElemB + uiGridKernelIdxX);
            pBlockSharedB[uiSharedBlockIdx1d] = (
                ((!bInsideB) || (uiBIdxY>=uiM))
                ? static_cast<TElem>(0)
                : B[uiBIdx1d]);

            // Synchronize to make sure the sub-matrices are loaded before starting the computation.
            acc.syncBlockKernels();

            // Not really necessary because we wrote zeros into those cells.
            //if(bInsideC)
            //{
                // Dyadic product within shared memory.
                for(TIndex k(0); k < uiBlockKernelsExtent; ++k)
                {
                    fCSum += pBlockSharedA[uiBlockKernelIdxY*uiBlockKernelsExtentX + k]
                        * pBlockSharedB[k*uiBlockKernelsExtentY + uiBlockKernelIdxX];
                }
            //}

            // Synchronize to make sure that the preceding computation is done before loading two new sub-matrices of A and B in the next iteration.
            acc.syncBlockKernels();
        }
        
        // If the element is outside of the matrix it was only a helper thread that did not calculate any meaningful results
        if(bInsideC)
        {
            C[uiGridKernelIdxY*uiPitchElemC + uiGridKernelIdxX] += fCSum;
        }
    }
};

namespace alpaka
{
    //#############################################################################
    //! The trait for getting the size of the block shared extern memory for a kernel.
    //#############################################################################
    template<
        typename TAcc>
    struct BlockSharedExternMemSizeBytes<
        MatMulKernel<TAcc>>
    {
        //-----------------------------------------------------------------------------
        //! \return The size of the shared memory allocated for a block.
        //-----------------------------------------------------------------------------
        template<
            typename TElem,
            typename TIndex>
        ALPAKA_FCT_HOST static std::size_t getBlockSharedExternMemSizeBytes(
            alpaka::Vec<3u> const & v3uiBlockKernelsExtents,
            TIndex const &,
            TIndex const &,
            TIndex const &,
            TElem const * const,
            TIndex const &,
            TElem const * const,
            TIndex const &,
            TElem * const,
            TIndex const &)
        {
            // Reserve the buffer for the two blocks of A and B.
            return 2u * v3uiBlockKernelsExtents.prod() * sizeof(TElem);
        }
    };
}

//-----------------------------------------------------------------------------
//! Profiles the given kernel.
//-----------------------------------------------------------------------------
template<
    typename TExec,
    typename TStream,
    typename... TArgs>
void profileAcceleratedKernel(
    TExec const & exec,
    TStream const & stream, // \TODO: Add a getStream Method to the kernel executor and do not require this parameter!
    TArgs && ... args)
{
    std::cout
        << "profileAcceleratedKernel("
        << " kernelExecutor: " << typeid(TExec).name()
        << ")" << std::endl;

    auto const tpStart(std::chrono::high_resolution_clock::now());

    // Execute the accelerated kernel.
    exec(std::forward<TArgs>(args)...);
    
    // Wait for the stream to finish the kernel execution to measure its run time.
    alpaka::wait::wait(stream);

    auto const tpEnd(std::chrono::high_resolution_clock::now());

    auto const durElapsed(tpEnd - tpStart);

    std::cout << "Execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(durElapsed).count() << " ms" << std::endl;
}

//-----------------------------------------------------------------------------
//! Profiles the example kernel and checks the result.
//-----------------------------------------------------------------------------
struct MatMulTester
{
    template<
        typename TAcc>
    void operator()(
        TAcc const &,
        std::size_t const & uiL,
        std::size_t const & uiM,
        std::size_t const & uiN,
        bool const & bAdaptiveBlockKernelsExtent)
    {
        std::cout << std::endl;
        std::cout << "################################################################################" << std::endl;

        using Kernel = MatMulKernel<>;
        
        alpaka::Vec<2u> const v2uiExtentsA(
            static_cast<alpaka::Vec<2u>::Val>(uiM),
            static_cast<alpaka::Vec<2u>::Val>(uiL)
        );

        alpaka::Vec<2u> const v2uiExtentsB(
            static_cast<alpaka::Vec<2u>::Val>(uiN),
            static_cast<alpaka::Vec<2u>::Val>(uiM)
        );

        // Result matrix is LxN. We create one worker per result matrix cell.
        alpaka::Vec<2u> const v2uiExtentsC(
            static_cast<alpaka::Vec<2u>::Val>(uiN),
            static_cast<alpaka::Vec<2u>::Val>(uiL)
        );
        
        // Let alpaka calculate good block and grid sizes given our full problem extents.
        alpaka::Vec<3u> v3uiGridKernels(static_cast<alpaka::Vec<3u>::Val>(uiN), static_cast<alpaka::Vec<3u>::Val>(uiL), static_cast<alpaka::Vec<3u>::Val>(1u));
        alpaka::workdiv::BasicWorkDiv workDiv(
            bAdaptiveBlockKernelsExtent
            ? alpaka::workdiv::getValidWorkDiv<boost::mpl::vector<TAcc>>(v3uiGridKernels, false)
            : alpaka::workdiv::getValidWorkDiv<alpaka::acc::EnabledAccelerators>(v3uiGridKernels, false));
        // Assure that the extents are square.
        auto const uiMinExtent(std::min(workDiv.m_v3uiBlockKernelsExtents[0u], workDiv.m_v3uiBlockKernelsExtents[1u]));
        workDiv.m_v3uiGridBlocksExtents[0u] = static_cast<alpaka::Vec<3u>::Val>(std::ceil(static_cast<double>(uiN) / static_cast<double>(uiMinExtent)));
        workDiv.m_v3uiBlockKernelsExtents[0u] = uiMinExtent;
        workDiv.m_v3uiGridBlocksExtents[1u] = static_cast<alpaka::Vec<3u>::Val>(std::ceil(static_cast<double>(uiL) / static_cast<double>(uiMinExtent)));
        workDiv.m_v3uiBlockKernelsExtents[1u] = uiMinExtent;

        std::cout
            << "profileAcceleratedMatMulKernel("
            << " uiL:" << uiL
            << " uiM:" << uiM
            << " uiN:" << uiN
            << ", accelerator: " << alpaka::acc::getAccName<TAcc>()
            << ", kernel: " << typeid(Kernel).name()
            << ", workDiv: " << workDiv
            << ")" << std::endl;

        // Allocate the A and B matrices as st::vectors because this allows them to be filled with uint32_t(1).
        // alpaka::mem::set only supports setting all bytes leading to a value of 16843009 in all elements.
        std::vector<std::uint32_t> vuiA(uiL * uiM, 1u);
        std::vector<std::uint32_t> vuiB(uiM * uiN, 1u);
        // Wrap the std::vectors into a memory buffer object.
        // For 1D data this would not be required because alpaka::mem::copy is specialized for std::vector and std::array.
        // For multi dimensional data you could directly create them using alpaka::mem::alloc<Type, MemSpaceHost>, which is not used here.
        // Instead we use MemBufBasePlainPtrWrapper to wrap the data.
        using MemBufWrapper = alpaka::mem::MemBufBasePlainPtrWrapper<
            alpaka::mem::MemSpaceHost,
            std::uint32_t,
            alpaka::dim::Dim2>;
        MemBufWrapper memBufAHost(vuiA.data(), v2uiExtentsA);
        MemBufWrapper memBufBHost(vuiB.data(), v2uiExtentsB);

        // Allocate C and set it to zero.
        auto memBufCHost(alpaka::mem::alloc<std::uint32_t, alpaka::mem::MemSpaceHost>(v2uiExtentsC));
        alpaka::mem::set(memBufCHost, 0u, v2uiExtentsC);

        // Allocate the buffers on the accelerator.
        using AccMemSpace = typename alpaka::mem::MemSpaceT<TAcc>;
        auto memBufAAcc(alpaka::mem::alloc<std::uint32_t, AccMemSpace>(v2uiExtentsA));
        auto memBufBAcc(alpaka::mem::alloc<std::uint32_t, AccMemSpace>(v2uiExtentsB));
        auto memBufCAcc(alpaka::mem::alloc<std::uint32_t, AccMemSpace>(v2uiExtentsC));

        // Get a new stream.
        alpaka::stream::StreamT<TAcc> stream;

        // Copy Host -> Acc.
        alpaka::mem::copy(memBufAAcc, memBufAHost, v2uiExtentsA, stream);
        alpaka::mem::copy(memBufBAcc, memBufBHost, v2uiExtentsB, stream);
        alpaka::mem::copy(memBufCAcc, memBufCHost, v2uiExtentsC, stream);

        // Build the kernel executor.
        auto exec(alpaka::createKernelExecutor<TAcc, Kernel>());
        // Profile the kernel execution.
        profileAcceleratedKernel(exec(workDiv, stream),
            stream,
            static_cast<std::uint32_t>(uiL),
            static_cast<std::uint32_t>(uiM),
            static_cast<std::uint32_t>(uiN),
            alpaka::mem::getNativePtr(memBufAAcc),
            static_cast<std::uint32_t>(alpaka::mem::getPitchElements(memBufAAcc)),
            alpaka::mem::getNativePtr(memBufBAcc),
            static_cast<std::uint32_t>(alpaka::mem::getPitchElements(memBufBAcc)),
            alpaka::mem::getNativePtr(memBufCAcc),
            static_cast<std::uint32_t>(alpaka::mem::getPitchElements(memBufCAcc)));

        // Copy back the result.
        alpaka::mem::copy(memBufCHost, memBufCAcc, v2uiExtentsC, stream);
        
        // Wait for the stream to finish the memory operation.
        alpaka::wait::wait(stream);

        // Assert that the results are correct.
        // When multiplying square matrices filled with ones, the result of each cell is the size of the matrix.
        std::uint32_t const uiCorrectResult(static_cast<std::uint32_t>(uiM));

        bool bResultCorrect(true);
        auto const pHostData(alpaka::mem::getNativePtr(memBufCHost));
        for(std::size_t i(0u);
            i < uiL * uiN;
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
int main(
    int argc,
    char * argv[])
{
    try
    {
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
        // Declare the supported options.
        boost::program_options::options_description desc("Available options");
        desc.add_options()
            ("help",
            "Prints the help message.")
            ("adaptiveBlockKernelsExtent,a",
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
            alpaka::acc::writeEnabledAccelerators(std::cout);

            std::cout << std::endl;

#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
            assert(vm.count("adaptiveBlockKernelsExtent") > 0);
            bool const bAdaptiveBlockKernelsExtent(vm["adaptiveBlockKernelsExtent"].as<bool>());
#else
            bool const bAdaptiveBlockKernelsExtent(true);
#endif
            std::cout << "Adaptive block kernel size:" << bAdaptiveBlockKernelsExtent << std::endl;

#ifdef ALPAKA_CUDA_ENABLED
            // Select the first CUDA device.
            // NOTE: This is not required to run any kernels on the CUDA accelerator because all accelerators have a default device. This only shows the possibility.
            alpaka::dev::DevManT<alpaka::AccCuda>::setCurrentDevice(
                alpaka::dev::DevManT<alpaka::AccCuda>::getCurrentDevice());
#endif
            MatMulTester matMulTester;

            // For different matrix sizes.
#if ALPAKA_INTEGRATION_TEST
            for(std::size_t uiL(1u); uiL <= 64u; uiL *= 8u)
            {
                for(std::size_t uiM(1u); uiM <= 512u; uiM *= 8u)
                {
                    for(std::size_t uiN(1u); uiN <= 64u; uiN *= 8u)
                    {
#else
            for(std::size_t uiL(1u); uiL <= 1024u; uiL *= 4u)
            {
                for(std::size_t uiM(1u); uiM <= 1024u; uiM *= 4u)
                {
                    for(std::size_t uiN(1u); uiN <= 1024u; uiN *= 4u)
                    {
#endif
                        std::cout << std::endl;

                        // Execute the kernel on all enabled accelerators.
                        boost::mpl::for_each<alpaka::acc::EnabledAccelerators>(
                            std::bind(
                                matMulTester,
                                std::placeholders::_1,
                                uiL, uiM, uiN,
                                bAdaptiveBlockKernelsExtent));
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
