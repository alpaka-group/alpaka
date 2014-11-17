/**
* Copyright 2014 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of either the GNU General Public License or
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
#include <alpaka/alpaka.hpp>    // alpaka::buildKernelExecutor<...>
#include <alpaka/WorkSize.hpp>  // alpaka::WorkSizeDefault

#include <chrono>               // std::chrono::high_resolution_clock
#include <cassert>              // assert
#include <iostream>             // std::cout
#include <vector>               // std::vector
#include <typeinfo>             // typeid
#include <utility>              // std::forward

#ifdef ALPAKA_CUDA_ENABLED
    #include <cuda.h>
#endif

//#############################################################################
//! A matrix multiplication kernel.
//! Computes C += A*B.
//! This is a adaption of the algorithm from the CUDA developers guide.
//! \param TAcc The accelerator environment to be executed on.
//#############################################################################
template<typename TAcc = boost::mpl::_1>
class MatMulKernel :
    public alpaka::IAcc<TAcc>
{
public:
    //-----------------------------------------------------------------------------
    //! The kernel.
    //-----------------------------------------------------------------------------
    template <typename TElement>
    ALPAKA_FCT_CPU_CUDA void operator()(
        std::uint32_t const n,
        TElement const * const A,
        TElement const * const B,
        TElement * const C) const
    {
        // Column and row of C to calculate.
        auto const v2uiGridKernelIdx(getIdx<alpaka::Grid, alpaka::Kernels>().subvec<2u>());
        auto const & cx(v2uiGridKernelIdx[0]);
        auto const & cy(v2uiGridKernelIdx[1]);
        // Column and row inside the block of C to calculate.
        auto const v2uiBlockKernelIdx(getIdx<alpaka::Block, alpaka::Kernels>().subvec<2u>());
        auto const & col(v2uiBlockKernelIdx[0]);
        auto const & row(v2uiBlockKernelIdx[1]);

        auto const v2uiBlockSize(getSize<alpaka::Block, alpaka::Kernels>().subvec<2u>());
        auto const & uiBlockSizeX(v2uiBlockSize[0]);
        auto const & uiBlockSizeY(v2uiBlockSize[1]);
        auto const uiBlockSizeLin(uiBlockSizeX*uiBlockSizeY);

        // Shared memory used to store the current blocks of A and B.
        auto * const pBlockSharedA(getBlockSharedExternMem<TElement>());
        auto * const pBlockSharedB(pBlockSharedA + uiBlockSizeLin);

        TElement fCSum(0);

        // If the element is outside of the matrix, write zero into the shared block and prevent out-of-bounds access to A and B 
        bool const bOutsideMatrix((cx>=n) || (cy>=n));

        // Loop over all blocks of A and B that are required to compute the C block. 
        auto const uiGridSizeX(getSize<alpaka::Grid, alpaka::Blocks>()[0]);
        for(std::uint32_t l(0); l<uiGridSizeX; ++l)
        {
            // Copy data to shared memory.
            auto const uiIndexA(cy*n + l*uiBlockSizeX + col);
            pBlockSharedA[row*uiBlockSizeX + col] = bOutsideMatrix ? 0 : A[uiIndexA];
            auto const uiIndexB((l*uiBlockSizeX+row)*n + cx);
            pBlockSharedB[row*uiBlockSizeX + col] = bOutsideMatrix ? 0 : B[uiIndexB];

            // Synchronize to make sure the sub-matrices are loaded before starting the computation.
            syncBlockKernels();

            // Dyadic product within shared memory.
            for(std::uint32_t k(0); k<uiBlockSizeY; ++k)
            {
                fCSum += pBlockSharedA[row*uiBlockSizeX + k] * pBlockSharedB[k*uiBlockSizeX + col];
            }

            // Synchronize to make sure that the preceding computation is done before loading two new sub-matrices of A and B in the next iteration.
            syncBlockKernels();
        }

        if(!bOutsideMatrix)
        {
            C[cy*n + cx] += fCSum;
        }
    }
};

namespace alpaka
{
    //#############################################################################
    //! The trait for getting the size of the block shared extern memory for a kernel.
    //#############################################################################
    template<typename TAcc>
    struct BlockSharedExternMemSizeBytes<MatMulKernel<TAcc>>
    {
        //-----------------------------------------------------------------------------
        //! \return The size of the shared memory allocated for a block.
        //-----------------------------------------------------------------------------
        template<typename TElement>
        static std::size_t getBlockSharedExternMemSizeBytes(
            alpaka::vec<3u> const & v3uiSizeBlockKernels,
            std::uint32_t const ,
            TElement const * const ,
            TElement const * const ,
            TElement * const )
        {
            // Reserve the buffer for the two blocks of A and B.
            return 2u * v3uiSizeBlockKernels.prod() * sizeof(TElement);
        }
    };
}

//-----------------------------------------------------------------------------
//! Profiles the given kernel.
//-----------------------------------------------------------------------------
template<typename TExec, typename TWorkSize, typename... TArgs>
void profileAcceleratedKernel(TExec const & exec, alpaka::IWorkSize<TWorkSize> const & workSize, TArgs && ... args)
{
    std::cout
        << "profileAcceleratedKernel("
        << " kernelExecutor: " << typeid(TExec).name()
        << ", workSize: " << workSize
        << ")" << std::endl;

    auto const tpStart(std::chrono::high_resolution_clock::now());

    // Execute the accelerated kernel.
    exec(workSize, std::forward<TArgs>(args)...);

    auto const tpEnd(std::chrono::high_resolution_clock::now());

    auto const durElapsed(tpEnd - tpStart);

    std::cout << "Execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(durElapsed).count() << " ms" << std::endl;
}

//-----------------------------------------------------------------------------
//! Profiles the given kernel (default Version).
//-----------------------------------------------------------------------------
template<typename TAcc>
class AcceleratedMatMulKernelProfiler
{
public:
    template<typename TKernel, typename TWorkSize, typename... TKernelConstrArgs>
    void operator()(alpaka::IWorkSize<TWorkSize> const & workSize, std::uint32_t const uiMatrixSize, std::uint32_t * const pA, std::uint32_t * const pB, std::uint32_t * const pC, TKernelConstrArgs && ... args)
    {
        std::cout
            << "AcceleratedExampleKernelProfiler("
            << " accelerator: " << typeid(TAcc).name()
            << ", kernel: " << typeid(TKernel).name()
            << ")" << std::endl;

        auto exec(alpaka::buildKernelExecutor<TAcc, TKernel>(std::forward<TKernelConstrArgs>(args)...));
        profileAcceleratedKernel(exec, workSize, uiMatrixSize, pA, pB, pC);
    }
};

#ifdef ALPAKA_CUDA_ENABLED
//-----------------------------------------------------------------------------
//! Profiles the given kernel (specialized for CUDA).
//-----------------------------------------------------------------------------
template<>
class AcceleratedMatMulKernelProfiler<alpaka::AccCuda>
{
public:
    template<typename TKernel, typename TWorkSize, typename... TKernelConstrArgs>
    void operator()(alpaka::IWorkSize<TWorkSize> const & workSize, std::uint32_t const uiMatrixSize, std::uint32_t * const pA, std::uint32_t * const pB, std::uint32_t * const pC, TKernelConstrArgs && ... args)
    {
        std::cout
            << "AcceleratedExampleKernelProfiler("
            << " accelerator: " << typeid(TAcc).name()
            << ", kernel: " << typeid(TKernel).name()
            << ")" << std::endl;

        std::size_t const uiNumBlocksInGrid(workSize.template getSize<alpaka::Grid, alpaka::Blocks, alpaka::Linear>());

        std::uint32_t * pBlockRetValsDev (nullptr);
        std::size_t const uiNumBytes(uiNumBlocksInGrid * sizeof(std::uint32_t));
        ALPAKA_CUDA_CHECK(cudaMalloc((void **) &pBlockRetValsDev, uiNumBytes));
        ALPAKA_CUDA_CHECK(cudaMemcpy(pBlockRetValsDev, puiBlockRetVals, uiNumBytes, cudaMemcpyHostToDevice));

        auto exec(alpaka::buildKernelExecutor<TAcc, TKernel>(std::forward<TKernelConstrArgs>(args)...));
        profileAcceleratedKernel(exec, workSize, uiMatrixSize, pA, pB, pC);
        
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
void profileAcceleratedMatMulKernel(alpaka::IWorkSize<TWorkSize> const & workSize, std::uint32_t const & uiMatrixSize)
{
    std::cout
        << "profileAcceleratedMatMulKernel("
        << " uiMatrixSize:" << uiMatrixSize
        << ")" << std::endl;

    std::vector<std::uint32_t> vuiA(uiMatrixSize*uiMatrixSize, 1);
    std::vector<std::uint32_t> vuiB(uiMatrixSize*uiMatrixSize, 1);
    std::vector<std::uint32_t> vuiC(uiMatrixSize*uiMatrixSize, 0);

    AcceleratedMatMulKernelProfiler<TAcc>().operator()<MatMulKernel<>>(workSize, uiMatrixSize, vuiA.data(), vuiB.data(), vuiC.data());

    // Assert that the results are correct.
    std::uint32_t const uiCorrectResult(uiMatrixSize);

    bool bResultCorrect(true);
    for(std::size_t i(0); i<uiMatrixSize*uiMatrixSize; ++i)
    {
        if(vuiC[i] != uiCorrectResult)
        {
            std::cout << "C[" << i << "] == " << vuiC[i] << " != " << uiCorrectResult << std::endl;
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
        std::cout << "                              alpaka matmul test                                " << std::endl;
        std::cout << "################################################################################" << std::endl;
        std::cout << std::endl;

        // Logs the enabled accelerators.
        logEnabledAccelerators();

        std::cout << std::endl;
        // Initialize the accelerators.
        initAccelerators();

        for(std::uint32_t uiMatrixSize(16u); uiMatrixSize<1024u; uiMatrixSize *= 2u)
        {
            // Set the block size (to the minimum all enabled tests support).
            alpaka::vec<3u> const v3uiSizeBlockKernels(
#if defined ALPAKA_SERIAL_ENABLED
                1u, 1u, 1u
#elif defined ALPAKA_OPENMP_ENABLED
                4u, 4u, 1u
#elif defined ALPAKA_CUDA_ENABLED || defined ALPAKA_THREADS_ENABLED || defined ALPAKA_FIBERS_ENABLED
                16u, 16u, 1u
#else
                1u, 1u, 1u
#endif
                );

            // Set the grid size.
            alpaka::vec<3u> const v3uiSizeGridBlocks(((uiMatrixSize-1u)/v3uiSizeBlockKernels[0u])+1u, ((uiMatrixSize-1u)/v3uiSizeBlockKernels[1u])+1u, 1u);

            alpaka::WorkSize const workSize(v3uiSizeGridBlocks, v3uiSizeBlockKernels);

#ifdef ALPAKA_SERIAL_ENABLED
            std::cout << std::endl;
            std::cout << "################################################################################" << std::endl;
            profileAcceleratedMatMulKernel<alpaka::AccSerial>(workSize, uiMatrixSize);
            std::cout << "################################################################################" << std::endl;
#endif
#ifdef ALPAKA_THREADS_ENABLED
            std::cout << std::endl;
            std::cout << "################################################################################" << std::endl;
            profileAcceleratedMatMulKernel<alpaka::AccThreads>(workSize, uiMatrixSize);
            std::cout << "################################################################################" << std::endl;
#endif
#ifdef ALPAKA_FIBERS_ENABLED
            std::cout << std::endl;
            std::cout << "################################################################################" << std::endl;
            profileAcceleratedMatMulKernel<alpaka::AccFibers>(workSize, uiMatrixSize);
            std::cout << "################################################################################" << std::endl;
#endif
#ifdef ALPAKA_OPENMP_ENABLED
            std::cout << std::endl;
            std::cout << "################################################################################" << std::endl;
            profileAcceleratedMatMulKernel<alpaka::AccOpenMp>(workSize, uiMatrixSize);
            std::cout << "################################################################################" << std::endl;
#endif
#ifdef ALPAKA_CUDA_ENABLED
            std::cout << std::endl;
            std::cout << "################################################################################" << std::endl;
            profileAcceleratedMatMulKernel<alpaka::AccCuda>(workSize, uiMatrixSize);
            std::cout << "################################################################################" << std::endl;
#endif
            std::cout << std::endl;
        }
        return 0;
    }
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