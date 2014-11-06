/**
* Copyright 2014 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of of either the GNU General Public License or
* the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License and the GNU Lesser General Public License
* for more details.
*
* You should have received a copy of the GNU General Public License
* and the GNU Lesser General Public License along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include <alpaka/IAcc.hpp>      // alpaka::IAcc<...>
#include <alpaka/Executor.hpp>  // alpaka::buildKernelExecutor<...>
#include <alpaka/WorkSize.hpp>  // alpaka::WorkSizeDefault

#include <chrono>               // std::chrono::high_resolution_clock
#include <cassert>              // assert
#include <iostream>             // std::cout
#include <vector>               // std::vector
#include <typeinfo>             // typeid

#ifdef ALPAKA_CUDA_ENABLED
    #include <cuda.h>
#endif

//#############################################################################
//! An accelerated test kernel.
//! Uses atomicFetchAdd(), syncBlockKernels(), shared memory, getIdx, getSize, global memory to compute a (useless) result.
//! \param TAcc The accelerator environment to be executed on.
//#############################################################################
template<typename TAcc = boost::mpl::_1>
class ExampleAcceleratedKernel :
	public alpaka::IAcc<TAcc>
{
public:
    //-----------------------------------------------------------------------------
    //! The kernel.
    //-----------------------------------------------------------------------------
    ALPAKA_FCT_CPU_CUDA void operator()(std::uint32_t * const puiBlockRetVals, std::uint32_t const uiNumUselessWork) const
    {
        //alpaka::vec<3> const uiIdxBlockKernels(getIdx<alpaka::Block, alpaka::Kernels>());
        //alpaka::vec<3> const v3uiSizeBlockKernels(getSize<alpaka::Block, alpaka::Kernels>());
        //alpaka::vec<3> const v3uiSizeGridBlocks(getSize<alpaka::Grid, alpaka::Blocks>());

        // The number of kernels in this block.
		std::uint32_t const uiNumKernelsInBlock(getSize<alpaka::Block, alpaka::Kernels, alpaka::Linear>());

        // Get the extern allocated shared memory.
        std::uint32_t * const pBlockShared(getBlockSharedExternMem<std::uint32_t>());

        //std::uint32_t * const pBlockShared1(getBlockSharedMem<std::uint32_t, 32>());
        //std::uint32_t * const pBlockShared2(getBlockSharedMem<std::uint32_t, 16>());

        // Calculate linearized index of the kernel in the block.
        std::uint32_t const uiIdxBlockKernelsLin(getIdx<alpaka::Block, alpaka::Kernels, alpaka::Linear>());

        // Fill the shared block with the kernel ids [1+X, 2+X, 3+X, ..., #Threads+X].
        std::uint32_t iSum1(uiIdxBlockKernelsLin+1);
        for(std::uint32_t i(0); i<uiNumUselessWork; ++i)
        {
            iSum1 += i;
        }
        pBlockShared[uiIdxBlockKernelsLin] = iSum1;

        // Synchronize all kernels because now we are writing to the memory again but inverse.
        syncBlockKernels();

        // Do something useless.
        std::uint32_t iSum2(uiIdxBlockKernelsLin);
        for(std::uint32_t i(0); i<uiNumUselessWork; ++i)
        {
            iSum2 -= i;
        }

        // Add the inverse so that every cell is filled with [#Kernels, #Kernels, ..., #Kernels].
        pBlockShared[(uiNumKernelsInBlock-1)-uiIdxBlockKernelsLin] += iSum2;

        // Synchronize all kernels again.
        syncBlockKernels();

        // Now add up all the cells atomically and write the result to cell 0 of the shared memory.
        if(uiIdxBlockKernelsLin > 0)
        {
            atomicFetchAdd(&pBlockShared[0], pBlockShared[uiIdxBlockKernelsLin]);
        }

        syncBlockKernels();

        // Only master writes result to global mem.
        if(uiIdxBlockKernelsLin==0)
        {
            //alpaka::vec<3> const blockIdx(TAcc::getIdxGridBlock());
            // Calculate linearized block id.
            std::uint32_t const bId(getIdx<alpaka::Grid, alpaka::Blocks, alpaka::Linear>());

            puiBlockRetVals[bId] = pBlockShared[0];
        }
    }

    //-----------------------------------------------------------------------------
    //! \return The size of the shared memory allocated for a block.
    //-----------------------------------------------------------------------------
    static std::size_t getBlockSharedMemSizeBytes(alpaka::vec<3> const v3uiSizeBlockKernels)
    {
        return v3uiSizeBlockKernels.prod() * sizeof(std::uint32_t);
    }
};

//-----------------------------------------------------------------------------
//! Profiles the given kernel.
//-----------------------------------------------------------------------------
template<typename TAcc, typename TKernel, typename TWorkSize, typename... TArgs>
void profileAcceleratedKernel(TKernel const & kernel, TWorkSize const & workSize, TArgs && ... args)
{
    std::cout
        << "profileAcceleratedKernel("
        << " kernel: " << typeid(TKernel).name()
        << ", accelerator: " << typeid(TAcc).name()
        << ", workSize: " << workSize << ")"
        << std::endl;

    auto const tpStart(std::chrono::high_resolution_clock::now());

    // Execute the accelerated kernel.
    alpaka::buildKernelExecutor<TAcc>(workSize, kernel)(std::forward<TArgs>(args)...);

    auto const tpEnd(std::chrono::high_resolution_clock::now());

    auto const durElapsed(tpEnd - tpStart);

    std::cout << "Execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(durElapsed).count() << " ms" << std::endl;
}

//-----------------------------------------------------------------------------
//! Profiles the example kernel (default CPU Version).
//-----------------------------------------------------------------------------
template<typename TAcc>
class AcceleratedExampleKernelProfiler
{
public:
    template<typename TWorkSize>
    void operator()(TWorkSize const & workSize, std::uint32_t * const puiBlockRetVals, std::uint32_t const uiNumUselessWork)
    {
        profileAcceleratedKernel<TAcc>(ExampleAcceleratedKernel<>(), workSize, puiBlockRetVals, uiNumUselessWork);
    }
};

#ifdef ALPAKA_CUDA_ENABLED
//-----------------------------------------------------------------------------
//! Profiles the example kernel (specialized for CUDA).
//-----------------------------------------------------------------------------
template<>
class AcceleratedExampleKernelProfiler<alpaka::AccCuda>
{
public:
    template<typename TWorkSize>
    void operator()(TWorkSize const & workSize, std::uint32_t * const puiBlockRetVals, std::uint32_t const uiNumUselessWork)
    {
        std::size_t const uiNumBlocksInGrid(workSize.template getSize<alpaka::Grid, alpaka::Blocks, alpaka::Linear>());

        std::uint32_t * pBlockRetValsDev (nullptr);
        std::size_t const uiNumBytes(uiNumBlocksInGrid * sizeof(std::uint32_t));
        ALPAKA_CUDA_CHECK(cudaMalloc((void **) &pBlockRetValsDev, uiNumBytes));
        ALPAKA_CUDA_CHECK(cudaMemcpy(pBlockRetValsDev, puiBlockRetVals, uiNumBytes, cudaMemcpyHostToDevice));

		profileAcceleratedKernel<alpaka::AccCuda>(ExampleAcceleratedKernel<>(), workSize, pBlockRetValsDev, uiNumUselessWork);
        
        ALPAKA_CUDA_CHECK(cudaDeviceSynchronize());

        ALPAKA_CUDA_CHECK(cudaMemcpy(puiBlockRetVals, pBlockRetValsDev, uiNumBytes, cudaMemcpyDeviceToHost));
        ALPAKA_CUDA_CHECK(cudaFree(pBlockRetValsDev)); 
    }
};
#endif

//-----------------------------------------------------------------------------
//! Profiles the example kernel and checks the result.
//-----------------------------------------------------------------------------
template<typename TAcc, typename TWorkSize>
void profileAcceleratedExampleKernel(TWorkSize const & workSize, std::uint32_t const uiNumUselessWork)
{
    std::size_t const uiNumBlocksInGrid(workSize.template getSize<alpaka::Grid, alpaka::Blocks, alpaka::Linear>());
    std::size_t const uiNumKernelsInBlock(workSize.template getSize<alpaka::Block, alpaka::Kernels, alpaka::Linear>());

    // An array for the return values calculated by the blocks.
    std::vector<std::uint32_t> vuiBlockRetVals(uiNumBlocksInGrid, 0);

    AcceleratedExampleKernelProfiler<TAcc>()(workSize, vuiBlockRetVals.data(), uiNumUselessWork);

    // Assert that the results are correct.
    bool bResultCorrect(true);
    for(std::size_t i(0); i<uiNumBlocksInGrid; ++i)
    {
        if(static_cast<std::size_t>(vuiBlockRetVals[i]) != uiNumKernelsInBlock*uiNumKernelsInBlock)
        {
            std::cout << "vuiBlockRetVals[" << i << "] == " << vuiBlockRetVals[i] << " != " << uiNumKernelsInBlock*uiNumKernelsInBlock << std::endl;
            bResultCorrect = false;
        }
    }

    if(bResultCorrect)
    {
        std::cout << "Execution results correct!" << std::endl;
    }
}

//-----------------------------------------------------------------------------
//! Logs the enabled accelerators.
//-----------------------------------------------------------------------------
void logEnabledAccelerators()
{
    std::cout << "Accelerators enabled: ";
#ifdef ALPAKA_SERIAL_ENABLED
    std::cout << "ALPAKA_SERIAL_ENABLED ";
#endif
#ifdef ALPAKA_THREADS_ENABLED
    std::cout << "ALPAKA_THREADS_ENABLED ";
#endif
#ifdef ALPAKA_FIBERS_ENABLED
    std::cout << "ALPAKA_FIBERS_ENABLED ";
#endif
#ifdef ALPAKA_OPENMP_ENABLED
    std::cout << "ALPAKA_OPENMP_ENABLED ";
#endif
#ifdef ALPAKA_CUDA_ENABLED
    std::cout << "ALPAKA_CUDA_ENABLED ";
#endif
    std::cout << std::endl;
}

//-----------------------------------------------------------------------------
//! Initializes the accelerators.
//-----------------------------------------------------------------------------
void initAccelerators()
{
#ifdef _DEBUG
    std::cout << "[+] initAccelerators()" << std::endl;
#endif

#ifdef ALPAKA_CUDA_ENABLED
    alpaka::AccCuda::setDevice(0);
#endif

#ifdef _DEBUG
    std::cout << "[-] initAccelerators()" << std::endl;
#endif
}
//-----------------------------------------------------------------------------
//! Program entry point.
//-----------------------------------------------------------------------------
int main()
{
    try
    {
        std::cout << std::endl;
        std::cout << "################################################################################" << std::endl;
        std::cout << "                              alpaka basic test                                 " << std::endl;
        std::cout << "################################################################################" << std::endl;
        std::cout << std::endl;

        // Logs the enabled accelerators.
        logEnabledAccelerators();

        std::cout << std::endl;
        // Initialize the accelerators.
        initAccelerators();

        // Set the grid size.
        alpaka::vec<3> const v3uiSizeGridBlocks(16u, 8u, 4u);

        // Set the block size (to the minimum all enabled tests support).
        alpaka::vec<3> const v3uiSizeBlockKernels(
#if defined ALPAKA_SERIAL_ENABLED
        1u, 1u, 1u
#elif defined ALPAKA_OPENMP_ENABLED
        4u, 4u, 2u
#elif defined ALPAKA_CUDA_ENABLED || defined ALPAKA_THREADS_ENABLED || defined ALPAKA_FIBERS_ENABLED
        16u, 16u, 2u
#else
        1u, 1u, 1u
#endif
        );

        std::uint32_t const uiNumUselessWork(1000);

        alpaka::WorkSize const workSize(v3uiSizeGridBlocks, v3uiSizeBlockKernels);

#ifdef ALPAKA_SERIAL_ENABLED
        std::cout << std::endl;
        std::cout << "################################################################################" << std::endl;
		profileAcceleratedExampleKernel<alpaka::AccSerial>(workSize, uiNumUselessWork);
        std::cout << "################################################################################" << std::endl;
#endif
#ifdef ALPAKA_THREADS_ENABLED
        std::cout << std::endl;
        std::cout << "################################################################################" << std::endl;
		profileAcceleratedExampleKernel<alpaka::AccThreads>(workSize, uiNumUselessWork);
        std::cout << "################################################################################" << std::endl;
#endif
#ifdef ALPAKA_FIBERS_ENABLED
        std::cout << std::endl;
        std::cout << "################################################################################" << std::endl;
		profileAcceleratedExampleKernel<alpaka::AccFibers>(workSize, uiNumUselessWork);
        std::cout << "################################################################################" << std::endl;
#endif
#ifdef ALPAKA_OPENMP_ENABLED
        std::cout << std::endl;
        std::cout << "################################################################################" << std::endl;
		profileAcceleratedExampleKernel<alpaka::AccOpenMp>(workSize, uiNumUselessWork);
        std::cout << "################################################################################" << std::endl;
#endif
#ifdef ALPAKA_CUDA_ENABLED
        std::cout << std::endl;
        std::cout << "################################################################################" << std::endl;
		profileAcceleratedExampleKernel<alpaka::AccCuda>(workSize, uiNumUselessWork);
        std::cout << "################################################################################" << std::endl;
#endif
        std::cout << std::endl;

        return 0;
    }
    /*catch(boost::exception const & e)
    {
        std::cerr << boost::diagnostic_information(e) << std::endl;
        return 1;
    }*/
    catch(std::exception const & e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    catch(...)
    {
        std::cerr << "Unknown Exception" << std::endl;
        return 1;
    }
}