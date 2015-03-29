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

#include <chrono>                           // std::chrono::high_resolution_clock
#include <cassert>                          // assert
#include <iostream>                         // std::cout
#include <typeinfo>                         // typeid
#include <utility>                          // std::forward

//#############################################################################
//! A vector addition kernel.
//! \tparam TAcc The accelerator environment to be executed on.
//#############################################################################
class VectorAddKernel
{
public:
    //-----------------------------------------------------------------------------
    //! The kernel entry point.
    //!
    //! \tparam TElem The matrix element type.
    //! \param acc The accelerator to be executed on.
    //! \param A The first source vector.
    //! \param B The second source vector.
    //! \param C The destination vector.
    //! \param uiNumElements The number of elements.
    //-----------------------------------------------------------------------------
    template<
        typename TAcc,
        typename TElem>
    ALPAKA_FCT_ACC auto operator()(
        TAcc const & acc, 
        TElem const * const A, 
        TElem const * const B, 
        TElem * const C, 
        std::size_t const & uiNumElements) const
    -> void
    {
        auto const uiGridThreadIdxX(acc.template getIdx<alpaka::Grid, alpaka::Threads>()[0u]);

        if (uiGridThreadIdxX < uiNumElements)
        {
            C[uiGridThreadIdxX] = A[uiGridThreadIdxX] + B[uiGridThreadIdxX];
        }
    }
};

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

//-----------------------------------------------------------------------------
//! Profiles the vector addition kernel.
//-----------------------------------------------------------------------------
struct VectorAddKernelTester
{
    template<
        typename TAcc>
    auto operator()(
        std::size_t const & uiNumElements)
    -> void
    {
        std::cout << std::endl;
        std::cout << "################################################################################" << std::endl;

        // Create the kernel functor.
        VectorAddKernel kernel;
        
        alpaka::Vec<1u> const v1uiExtents(
            static_cast<alpaka::Vec<1u>::Val>(uiNumElements)
        );

        // Let alpaka calculate good block and grid sizes given our full problem extents.
        alpaka::Vec<3u> v3uiGridThreads(static_cast<alpaka::Vec<3u>::Val>(uiNumElements), static_cast<alpaka::Vec<3u>::Val>(1u), static_cast<alpaka::Vec<3u>::Val>(1u));
        alpaka::workdiv::BasicWorkDiv const workDiv(alpaka::workdiv::getValidWorkDiv<boost::mpl::vector<TAcc>>(v3uiGridThreads, false));

        std::cout
            << "VectorAddKernelTester("
            << " uiNumElements:" << uiNumElements
            << ", accelerator: " << alpaka::acc::getAccName<TAcc>()
            << ", kernel: " << typeid(kernel).name()
            << ", workDiv: " << workDiv
            << ")" << std::endl;

        // Allocate host memory buffers.
        auto memBufHostA(alpaka::mem::alloc<float, alpaka::mem::SpaceHost>(v1uiExtents));
        auto memBufHostB(alpaka::mem::alloc<float, alpaka::mem::SpaceHost>(v1uiExtents));
        auto memBufHostC(alpaka::mem::alloc<float, alpaka::mem::SpaceHost>(v1uiExtents));
        
        // Initialize the host input vectors
        for (std::size_t i(0); i < uiNumElements; ++i)
        {
            alpaka::mem::getNativePtr(memBufHostA)[i] = static_cast<float>(rand())/static_cast<float>(RAND_MAX);
            alpaka::mem::getNativePtr(memBufHostB)[i] = static_cast<float>(rand())/static_cast<float>(RAND_MAX);
        }

        // Allocate the buffer on the accelerator.
        using AccMemSpace = typename alpaka::mem::SpaceT<TAcc>;
        auto memBufAccA(alpaka::mem::alloc<float, AccMemSpace>(v1uiExtents));
        auto memBufAccB(alpaka::mem::alloc<float, AccMemSpace>(v1uiExtents));
        auto memBufAccC(alpaka::mem::alloc<float, AccMemSpace>(v1uiExtents));

        // Get a new stream.
        alpaka::stream::StreamT<TAcc> stream;

        // Copy Host -> Acc.
        alpaka::mem::copy(memBufAccA, memBufHostA, v1uiExtents, stream);
        alpaka::mem::copy(memBufAccB, memBufHostB, v1uiExtents, stream);
        
        // Create the kernel executor.
        auto exec(alpaka::exec::create<TAcc>(workDiv, stream));
        // Profile the kernel execution.
        profileKernelExec(
            exec,
            kernel,
            alpaka::mem::getNativePtr(memBufAccA),
            alpaka::mem::getNativePtr(memBufAccB),
            alpaka::mem::getNativePtr(memBufAccC),
            static_cast<std::uint32_t>(uiNumElements));

        // Copy back the result.
        alpaka::mem::copy(memBufHostC, memBufAccC, v1uiExtents, stream);
        
        // Wait for the stream to finish the memory operation.
        alpaka::wait::wait(stream);
        
        bool bResultCorrect(true);
        auto const pHostData(alpaka::mem::getNativePtr(memBufHostC));
        for(std::size_t i(0u);
            i < uiNumElements;
            ++i)
        {
            auto const & uiVal(pHostData[i]);
            auto const uiCorrectResult(alpaka::mem::getNativePtr(memBufHostA)[i]+alpaka::mem::getNativePtr(memBufHostB)[i]);
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
        std::cout << std::endl;
        std::cout << "################################################################################" << std::endl;
        std::cout << "                            alpaka vector add test                              " << std::endl;
        std::cout << "################################################################################" << std::endl;
        std::cout << std::endl;

        // Logs the enabled accelerators.
        alpaka::acc::writeEnabledAccelerators(std::cout);

        std::cout << std::endl;

#ifdef ALPAKA_CUDA_ENABLED
        // Select the first CUDA device.
        // NOTE: This is not required to run any kernels on the CUDA accelerator because all accelerators have a default device. This only shows the possibility.
        alpaka::dev::DevManT<alpaka::AccCuda>::setCurrentDev(
            alpaka::dev::DevManT<alpaka::AccCuda>::getCurrentDev());
#endif
        VectorAddKernelTester vectorAddKernelTester;

        // For different sizes.
#if ALPAKA_INTEGRATION_TEST
        for(std::size_t uiSize(1u); uiSize <= 1u<<9u; uiSize *= 8u)
#else
        for(std::size_t uiSize(1u); uiSize <= 1u<<16u; uiSize *= 2u)
#endif
        {
            std::cout << std::endl;

            // Execute the kernel on all enabled accelerators.
            alpaka::ForEachType<alpaka::acc::EnabledAccelerators>(
                vectorAddKernelTester,
                uiSize);
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
