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

#include <alpaka/alpaka.hpp>        // alpaka::createKernelExecutor<...>

#include <chrono>                   // std::chrono::high_resolution_clock
#include <cassert>                  // assert
#include <iostream>                 // std::cout
#include <vector>                   // std::vector
#include <typeinfo>                 // typeid
#include <utility>                  // std::forward

#include <boost/mpl/for_each.hpp>   // boost::mpl::for_each

//#############################################################################
//! An accelerated test kernel.
//! Uses atomicOp(), syncBlockKernels(), shared memory, getIdx, getExtents, global memory to compute a (useless) result.
//! \tparam TAcc The accelerator environment to be executed on.
//! \tparam TuiNumUselessWork The number of useless calculations done in each kernel execution.
//#############################################################################
template<
    typename TuiNumUselessWork, 
    typename TAcc = alpaka::IAcc<>>
class SharedMemKernel
{
public:
    //-----------------------------------------------------------------------------
    //! Constructor.
    //-----------------------------------------------------------------------------
    SharedMemKernel(
        std::uint32_t const uiMult = 2) :
        m_uiMult(uiMult)
    {}

    //-----------------------------------------------------------------------------
    //! The kernel.
    //-----------------------------------------------------------------------------
    ALPAKA_FCT_ACC void operator()(
        TAcc const & acc,
        std::uint32_t * const puiBlockRetVals, 
        std::uint32_t const uiMult2) const
    {
        // The number of kernels in this block.
        std::size_t const uiNumKernelsInBlock(acc.template getWorkDiv<alpaka::Block, alpaka::Kernels, alpaka::dim::Dim1>()[0u]);

        // Get the extern allocated shared memory.
        std::uint32_t * const pBlockShared(acc.template getBlockSharedExternMem<std::uint32_t>());

        // Get some shared memory (allocate a second buffer directly afterwards to check for some synchronization bugs).
        //std::uint32_t * const pBlockShared1(acc.template allocBlockSharedMem<std::uint32_t, TuiNumUselessWork::value>());
        //std::uint32_t * const pBlockShared2(acc.template allocBlockSharedMem<std::uint32_t, TuiNumUselessWork::value>());

        // Calculate linearized index of the kernel in the block.
        std::size_t const uiIdxBlockKernelsLin(acc.template getIdx<alpaka::Block, alpaka::Kernels, alpaka::dim::Dim1>()[0u]);


        // Fill the shared block with the kernel ids [1+X, 2+X, 3+X, ..., #Threads+X].
        std::uint32_t iSum1(static_cast<std::uint32_t>(uiIdxBlockKernelsLin+1));
        for(std::uint32_t i(0); i<TuiNumUselessWork::value; ++i)
        {
            iSum1 += i;
        }
        pBlockShared[uiIdxBlockKernelsLin] = iSum1;


        // Synchronize all kernels because now we are writing to the memory again but inverse.
        acc.syncBlockKernels();

        // Do something useless.
        std::uint32_t iSum2(static_cast<std::uint32_t>(uiIdxBlockKernelsLin));
        for(std::uint32_t i(0); i<TuiNumUselessWork::value; ++i)
        {
            iSum2 -= i;
        }
        // Add the inverse so that every cell is filled with [#Kernels, #Kernels, ..., #Kernels].
        pBlockShared[(uiNumKernelsInBlock-1)-uiIdxBlockKernelsLin] += iSum2;


        // Synchronize all kernels again.
        acc.syncBlockKernels();

        // Now add up all the cells atomically and write the result to cell 0 of the shared memory.
        if(uiIdxBlockKernelsLin > 0)
        {
            acc.template atomicOp<alpaka::ops::Add>(&pBlockShared[0], pBlockShared[uiIdxBlockKernelsLin]);
        }


        acc.syncBlockKernels();

        // Only master writes result to global memory.
        if(uiIdxBlockKernelsLin==0)
        {
            // Calculate linearized block id.
            std::size_t const uiblockIdx(acc.template getIdx<alpaka::Grid, alpaka::Blocks, alpaka::dim::Dim1>()[0u]);

            puiBlockRetVals[uiblockIdx] = pBlockShared[0] * m_uiMult * uiMult2;
        }
    }

public:
    std::uint32_t /*const*/ m_uiMult;
};

namespace alpaka
{
    //#############################################################################
    //! The trait for getting the size of the block shared extern memory for a kernel.
    //#############################################################################
    template<
        class TuiNumUselessWork, 
        class TAcc>
    struct BlockSharedExternMemSizeBytes<
        SharedMemKernel<TuiNumUselessWork, TAcc>>
    {
        //-----------------------------------------------------------------------------
        //! \return The size of the shared memory allocated for a block.
        //-----------------------------------------------------------------------------
        template<typename... TArgs>
        ALPAKA_FCT_HOST static std::size_t getBlockSharedExternMemSizeBytes(
            alpaka::Vec<3u> const & v3uiBlockKernelsExtents, 
            TArgs && ...)
        {
            return v3uiBlockKernelsExtents.prod() * sizeof(std::uint32_t);
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
template<
    typename TuiNumUselessWork>
struct SharedMemTester
{
    template<
        typename TAcc, 
        typename TWorkDiv>
    void operator()(
        TAcc, 
        TWorkDiv const & workDiv, 
        std::uint32_t const uiMult2)
    {
        std::cout << std::endl;
        std::cout << "################################################################################" << std::endl;
        
        using Kernel = SharedMemKernel<TuiNumUselessWork>;
        using AccMemSpace = typename alpaka::mem::GetMemSpaceT<TAcc>;

        std::cout
            << "AcceleratedExampleKernelProfiler("
            << " accelerator: " << alpaka::acc::getAccName<TAcc>()
            << ", kernel: " << typeid(Kernel).name()
            << ", workDiv: " << workDiv
            << ")" << std::endl;

        std::size_t const uiGridBlocksCount(alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Blocks, alpaka::dim::Dim1>(workDiv)[0u]);
        std::size_t const uiBlockKernelsCount(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Kernels, alpaka::dim::Dim1>(workDiv)[0u]);

        // An array for the return values calculated by the blocks.
        std::vector<std::uint32_t> vuiBlockRetVals(uiGridBlocksCount, 0);

        // Allocate accelerator buffers and copy.
        std::size_t const uiSizeElements(uiGridBlocksCount);
        auto blockRetValsAcc(alpaka::mem::alloc<std::uint32_t, AccMemSpace>(uiSizeElements));
        alpaka::mem::copy(blockRetValsAcc, vuiBlockRetVals, uiSizeElements);

        std::uint32_t const m_uiMult(42);

        // Build the kernel executor.
        auto exec(alpaka::createKernelExecutor<TAcc, Kernel>(m_uiMult));
        // Get a new stream.
        alpaka::stream::GetStreamT<TAcc> stream;
        // Profile the kernel execution.
        profileAcceleratedKernel(
		    exec(workDiv, stream), 
		    stream,
		    alpaka::mem::getNativePtr(blockRetValsAcc),
		    uiMult2);

        // Copy back the result.
        alpaka::mem::copy(vuiBlockRetVals, blockRetValsAcc, uiSizeElements);

        // Assert that the results are correct.
        std::uint32_t const uiCorrectResult(static_cast<std::uint32_t>(uiBlockKernelsCount*uiBlockKernelsCount) * m_uiMult * uiMult2);

        bool bResultCorrect(true);
        for(std::size_t i(0); i<uiGridBlocksCount; ++i)
        {
            if(vuiBlockRetVals[i] != uiCorrectResult)
            {
                std::cout << "vuiBlockRetVals[" << i << "] == " << vuiBlockRetVals[i] << " != " << uiCorrectResult << std::endl;
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
int main()
{
    try
    {
        std::cout << std::endl;
        std::cout << "################################################################################" << std::endl;
        std::cout << "                            alpaka sharedMem test                               " << std::endl;
        std::cout << "################################################################################" << std::endl;
        std::cout << std::endl;

        // Logs the enabled accelerators.
        alpaka::acc::writeEnabledAccelerators(std::cout);

        std::cout << std::endl;
        
#ifdef ALPAKA_CUDA_ENABLED
        // Select the first CUDA device. 
        // NOTE: This is not required to run any kernels on the CUDA accelerator because all accelerators have a default device. This only shows the possibility.
        alpaka::dev::GetDevManT<alpaka::AccCuda>::setCurrentDevice(
            alpaka::dev::GetDevManT<alpaka::AccCuda>::getCurrentDevice());
#endif

        // Set the grid blocks extent.
        alpaka::workdiv::BasicWorkDiv const workDiv(
            alpaka::workdiv::getValidWorkDiv<alpaka::acc::EnabledAccelerators>(
                alpaka::Vec<3u>(16u, 8u, 4u)));

        using TuiNumUselessWork = boost::mpl::int_<100u>;
        std::uint32_t const uiMult2(5u);

        SharedMemTester<TuiNumUselessWork> sharedMemTester;
            
        // Execute the kernel on all enabled accelerators.
        boost::mpl::for_each<alpaka::acc::EnabledAccelerators>(
            std::bind(
                sharedMemTester, 
                std::placeholders::_1, 
                workDiv, 
                uiMult2));

        return sharedMemTester.bAllResultsCorrect ? EXIT_SUCCESS : EXIT_FAILURE;
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