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

#include <alpaka/alpaka.hpp>			// alpaka::createKernelExecutor<...>

#include <chrono>						// std::chrono::high_resolution_clock
#include <cassert>						// assert
#include <iostream>						// std::cout
#include <vector>						// std::vector
#include <typeinfo>						// typeid
#include <utility>						// std::forward
#include <functional>					// std::placeholders
#include <unordered_map>				// std::unordered_map

#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
#if BOOST_COMP_MSVC
 #pragma warning(push)
 #pragma warning(disable: 4512)			// boost/program_options/options_description.hpp(265): warning C4512: 'boost::program_options::options_description': assignment operator was implicitly defined as deleted
#endif

#include <boost/program_options.hpp>	// boost::program_options

#if BOOST_COMP_MSVC
 #pragma warning(pop)
#endif
#endif

#include <boost/mpl/for_each.hpp>       // boost::mpl::for_each

//#############################################################################
//! A matrix multiplication kernel.
//! Computes C += A*B.
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
    //! \param n The matrix size=width=height.
    //! \param A The pointer to the matrix A data.
    //! \param B The pointer to the matrix B data.
    //! \param C The pointer to the matrix C data.
    //-----------------------------------------------------------------------------
    template<
        typename TElem,
        typename TIndex>
    ALPAKA_FCT_ACC void operator()(
        TAcc const & acc,
        TIndex const & n,
        TElem const * const A,
        TIndex const & uiPitchElemA,
        TElem const * const B,
        TIndex const & uiPitchElemB,
        TElem * const C,
        TIndex const & uiPitchElemC) const
    {
        // Column and row of C to calculate.
        auto const v2uiGridKernelIdx(acc.template getIdx<alpaka::Grid, alpaka::Kernels>().template subvec<2u>());
        auto const & cx(v2uiGridKernelIdx[0u]);
        auto const & cy(v2uiGridKernelIdx[1u]);
        // Column and row inside the block of C to calculate.
        auto const v2uiBlockKernelIdx(acc.template getIdx<alpaka::Block, alpaka::Kernels>().template subvec<2u>());
        auto const & col(v2uiBlockKernelIdx[0u]);
        auto const & row(v2uiBlockKernelIdx[1u]);

        // The block kernels extents.
        auto const v2uiBlockKernelsExtent(acc.template getWorkDiv<alpaka::Block, alpaka::Kernels>().template subvec<2u>());
        auto const & uiBlockKernelsExtentX(v2uiBlockKernelsExtent[0u]);
        auto const & uiBlockKernelsExtentY(v2uiBlockKernelsExtent[1u]);
        auto const uiBlockKernelsExtent1d(uiBlockKernelsExtentX * uiBlockKernelsExtentY);

        // Shared memory used to store the current blocks of A and B.
        auto * const pBlockSharedA(acc.template getBlockSharedExternMem<TElem>());
        auto * const pBlockSharedB(pBlockSharedA + uiBlockKernelsExtent1d);

        // If the element is outside of the matrix, write zero into the shared block and prevent out-of-bounds access to A and B
        bool const bOutsideMatrix((cx >= n) || (cy >= n));

        auto const uiGridBlockExtentX(acc.template getWorkDiv<alpaka::Grid, alpaka::Blocks>()[0u]);

        auto const uiSharedBlockIdx1d(row * uiBlockKernelsExtentX + col);

        TElem fCSum(0);

        // Loop over all blocks of A and B that are required to compute the C block.
        for(TIndex l(0u); l < uiGridBlockExtentX; ++l)
        {
            // Copy data to shared memory.
            auto const uiIdxA(cy * uiPitchElemA + l * uiBlockKernelsExtentX + col);
            
            //printf("%lu %u ", uiIdxA, A[uiIdxA]);

            pBlockSharedA[uiSharedBlockIdx1d] = 
                bOutsideMatrix 
                ? 0
                : A[uiIdxA];
            auto const uiIdxB((l * uiBlockKernelsExtentX + row) * uiPitchElemB + cx);
            pBlockSharedB[uiSharedBlockIdx1d] = 
                bOutsideMatrix
                ? 0
                : B[uiIdxB];

            // Synchronize to make sure the sub-matrices are loaded before starting the computation.
            acc.syncBlockKernels();

            // Dyadic product within shared memory.
            for(TIndex k(0); k < uiBlockKernelsExtentY; ++k)
            {
                fCSum += pBlockSharedA[row * uiBlockKernelsExtentX + k]
                    * pBlockSharedB[k * uiBlockKernelsExtentX + col];
            }

            // Synchronize to make sure that the preceding computation is done before loading two new sub-matrices of A and B in the next iteration.
            acc.syncBlockKernels();
        }

        if(!bOutsideMatrix)
        {
            C[cy * uiPitchElemC + cx] += fCSum;
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
struct ProfileAcceleratedMatMulKernel
{
    template<
        typename TAcc>
    void operator()(
        TAcc,
        std::size_t const & uiMatrixSize,
        bool const & bAdaptiveBlockKernelsExtent)
    {
        std::cout << std::endl;
        std::cout << "################################################################################" << std::endl;

        using Kernel = MatMulKernel<>;

        // Let alpaka calculate good block and grid sizes given our full problem extents.
        alpaka::Vec<3u> v3uiGridKernels(static_cast<alpaka::Vec<3u>::Value>(uiMatrixSize), static_cast<alpaka::Vec<3u>::Value>(uiMatrixSize), static_cast<alpaka::Vec<3u>::Value>(1u));
        alpaka::workdiv::BasicWorkDiv const workDiv(
            bAdaptiveBlockKernelsExtent
            ? alpaka::workdiv::getValidWorkDiv<boost::mpl::vector<TAcc>>(v3uiGridKernels)
            : alpaka::workdiv::getValidWorkDiv<alpaka::acc::EnabledAccelerators>(v3uiGridKernels));

        std::cout
            << "profileAcceleratedMatMulKernel("
            << " uiMatrixSize:" << uiMatrixSize
            << ", accelerator: " << alpaka::acc::getAccName<TAcc>()
            << ", kernel: " << typeid(Kernel).name()
            << ", workDiv: " << workDiv
            << ")" << std::endl;

        alpaka::extent::BasicExtents<alpaka::dim::Dim2> const extents(
            uiMatrixSize,
            uiMatrixSize
        );

        // Allocate the A and B matrices as st::vectors because this allows them to be filled with uint32_t(1).
        // alpaka::mem::set only supports setting all bytes leading to a value of 16843009 in all elements.
        std::vector<std::uint32_t> vuiA(uiMatrixSize * uiMatrixSize, 1u);
        std::vector<std::uint32_t> vuiB(uiMatrixSize * uiMatrixSize, 1u);
        // Wrap the std::vectors into a memory buffer object.
        // For 1D data this would not be required because alpaka::mem::copy is specialized for std::vector and std::array.
        // For multi dimensional data you could directly create them using alpaka::mem::alloc<Type, MemSpaceHost>, which is not used here.
        // Instead we use MemBufPlainPtrWrapper to wrap the data.
        using MemBufWrapper = alpaka::mem::MemBufPlainPtrWrapper<
            alpaka::mem::MemSpaceHost,
            std::uint32_t,
            alpaka::dim::Dim2>;
        MemBufWrapper memBufAHost(vuiA.data(), extents);
        MemBufWrapper memBufBHost(vuiB.data(), extents);

        // Allocate C and set it to zero.
        auto memBufCHost(alpaka::mem::alloc<std::uint32_t, alpaka::mem::MemSpaceHost>(extents));
        alpaka::mem::set(memBufCHost, 0u, extents);

        // Allocate the buffers on the accelerator.
        using AccMemSpace = typename alpaka::mem::GetMemSpaceT<TAcc>;
        auto memBufAAcc(alpaka::mem::alloc<std::uint32_t, AccMemSpace>(extents));
        auto memBufBAcc(alpaka::mem::alloc<std::uint32_t, AccMemSpace>(extents));
        auto memBufCAcc(alpaka::mem::alloc<std::uint32_t, AccMemSpace>(extents));

        // Get a new stream.
        alpaka::stream::GetStreamT<TAcc> stream;

        // Copy Host -> Acc.
        alpaka::mem::copy(memBufAAcc, memBufAHost, extents, stream);
        alpaka::mem::copy(memBufBAcc, memBufBHost, extents, stream);
        alpaka::mem::copy(memBufCAcc, memBufCHost, extents, stream);

        // Build the kernel executor.
        auto exec(alpaka::createKernelExecutor<TAcc, Kernel>());
        // Profile the kernel execution.
        profileAcceleratedKernel(exec(workDiv, stream),
            stream,
            static_cast<std::uint32_t>(uiMatrixSize),
            alpaka::mem::getNativePtr(memBufAAcc),
            static_cast<std::uint32_t>(alpaka::mem::getPitchElements(memBufAAcc)),
            alpaka::mem::getNativePtr(memBufBAcc),
            static_cast<std::uint32_t>(alpaka::mem::getPitchElements(memBufBAcc)),
            alpaka::mem::getNativePtr(memBufCAcc),
            static_cast<std::uint32_t>(alpaka::mem::getPitchElements(memBufCAcc)));

        // Copy back the result.
        alpaka::mem::copy(memBufCHost, memBufCAcc, extents, stream);
        
        // Wait for the stream to finish the memory operation.
        alpaka::wait::wait(stream);

        // Assert that the results are correct.
        // When multiplying square matrices filled with ones, the result of each cell is the size of the matrix.
        std::uint32_t const uiCorrectResult(static_cast<std::uint32_t>(uiMatrixSize));

        bool bResultCorrect(true);
        auto const pHostData(alpaka::mem::getNativePtr(memBufCHost));
        for(std::size_t i(0u);
            i < uiMatrixSize * uiMatrixSize;
            ++i)
        {
            auto const uiVal(pHostData[i]);
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
    }
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
            std::cout << "                              alpaka matmul test                                " << std::endl;
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
            alpaka::dev::GetDevManT<alpaka::AccCuda>::setCurrentDevice(
                alpaka::dev::GetDevManT<alpaka::AccCuda>::getCurrentDevice());
#endif
            // For different matrix sizes.
            for(std::size_t uiMatrixSize(16u);
                uiMatrixSize < 1024u;
                uiMatrixSize *= 2u)
            {
                std::cout << std::endl;

                // Execute the kernel on all enabled accelerators.
                boost::mpl::for_each<alpaka::acc::EnabledAccelerators>(
                    std::bind(
                    ProfileAcceleratedMatMulKernel(),
                    std::placeholders::_1,
                    uiMatrixSize,
                    bAdaptiveBlockKernelsExtent)
                    );
            }
        }
        return EXIT_SUCCESS;
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
