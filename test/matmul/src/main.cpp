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

#include <alpaka/alpaka.hpp>    // alpaka::createKernelExecutor<...>

#include <chrono>               // std::chrono::high_resolution_clock
#include <cassert>              // assert
#include <iostream>             // std::cout
#include <vector>               // std::vector
#include <typeinfo>             // typeid
#include <utility>              // std::forward

//#############################################################################
//! A matrix multiplication kernel.
//! Computes C += A*B.
//! This is a adaption of the algorithm from the CUDA developers guide.
//! \tparam TAcc The accelerator environment to be executed on.
//#############################################################################
template<typename TAcc = alpaka::IAcc<>>
class MatMulKernel :
    public TAcc
{
public:
    //-----------------------------------------------------------------------------
    //! The kernel.
    //-----------------------------------------------------------------------------
    template <typename TElement>
    ALPAKA_FCT_HOST_ACC void operator()(
        std::uint32_t const n,
        TElement const * const A,
        TElement const * const B,
        TElement * const C) const
    {
        // Column and row of C to calculate.
        auto const v2uiGridKernelIdx(TAcc::template getIdx<alpaka::Grid, alpaka::Kernels>().template subvec<2u>());
        auto const & cx(v2uiGridKernelIdx[0]);
        auto const & cy(v2uiGridKernelIdx[1]);
        // Column and row inside the block of C to calculate.
        auto const v2uiBlockKernelIdx(TAcc::template getIdx<alpaka::Block, alpaka::Kernels>().template subvec<2u>());
        auto const & col(v2uiBlockKernelIdx[0]);
        auto const & row(v2uiBlockKernelIdx[1]);

        auto const v2uiBlockSize(TAcc::template getSize<alpaka::Block, alpaka::Kernels>().template subvec<2u>());
        auto const & uiBlockSizeX(v2uiBlockSize[0]);
        auto const & uiBlockSizeY(v2uiBlockSize[1]);
        auto const uiBlockSizeLin(uiBlockSizeX*uiBlockSizeY);

        // Shared memory used to store the current blocks of A and B.
        auto * const pBlockSharedA(TAcc::template getBlockSharedExternMem<TElement>());
        auto * const pBlockSharedB(pBlockSharedA + uiBlockSizeLin);

        TElement fCSum(0);

        // If the element is outside of the matrix, write zero into the shared block and prevent out-of-bounds access to A and B 
        bool const bOutsideMatrix((cx>=n) || (cy>=n));

        // Loop over all blocks of A and B that are required to compute the C block. 
        auto const uiGridSizeX(TAcc::template getSize<alpaka::Grid, alpaka::Blocks>()[0]);
        for(std::uint32_t l(0); l<uiGridSizeX; ++l)
        {
            // Copy data to shared memory.
            auto const uiIndexA(cy*n + l*uiBlockSizeX + col);
            pBlockSharedA[row*uiBlockSizeX + col] = bOutsideMatrix ? 0 : A[uiIndexA];
            auto const uiIndexB((l*uiBlockSizeX+row)*n + cx);
            pBlockSharedB[row*uiBlockSizeX + col] = bOutsideMatrix ? 0 : B[uiIndexB];

            // Synchronize to make sure the sub-matrices are loaded before starting the computation.
            TAcc::syncBlockKernels();

            // Dyadic product within shared memory.
            for(std::uint32_t k(0); k<uiBlockSizeY; ++k)
            {
                fCSum += pBlockSharedA[row*uiBlockSizeX + k] * pBlockSharedB[k*uiBlockSizeX + col];
            }

            // Synchronize to make sure that the preceding computation is done before loading two new sub-matrices of A and B in the next iteration.
            TAcc::syncBlockKernels();
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
template<typename TExec, typename... TArgs>
void profileAcceleratedKernel(TExec const & exec, TArgs && ... args)
{
    std::cout
        << "profileAcceleratedKernel("
        << " kernelExecutor: " << typeid(TExec).name()
        << ")" << std::endl;

    auto const tpStart(std::chrono::high_resolution_clock::now());

    // Execute the accelerated kernel.
    exec(std::forward<TArgs>(args)...);

    // Enqueue an event to wait for. This allows synchronization after the (possibly) asynchronous kernel execution.
    alpaka::event::Event<typename TExec::TAcc> ev;
    alpaka::event::eventEnqueue(ev);
    alpaka::event::eventWait(ev);

    auto const tpEnd(std::chrono::high_resolution_clock::now());

    auto const durElapsed(tpEnd - tpStart);

    std::cout << "Execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(durElapsed).count() << " ms" << std::endl;
}

//-----------------------------------------------------------------------------
//! Profiles the example kernel and checks the result.
//-----------------------------------------------------------------------------
template<typename TAcc, typename TWorkSize>
void profileAcceleratedMatMulKernel(alpaka::IWorkSize<TWorkSize> const & workSize, std::uint32_t const & uiMatrixSize)
{
    using TKernel = MatMulKernel<>;
    using TAccMemorySpace = typename TAcc::MemorySpace;

    std::cout
        << "profileAcceleratedMatMulKernel("
        << " uiMatrixSize:" << uiMatrixSize
        << ", accelerator: " << typeid(TAcc).name()
        << ", kernel: " << typeid(TKernel).name()
        << ", workSize: " << workSize
        << ")" << std::endl;

    // Initialize matrices.
    std::vector<std::uint32_t> vuiA(uiMatrixSize*uiMatrixSize, 1);
    std::vector<std::uint32_t> vuiB(uiMatrixSize*uiMatrixSize, 1);
    std::vector<std::uint32_t> vuiC(uiMatrixSize*uiMatrixSize, 0);

    // Allocate accelerator buffers and copy.
    std::size_t const uiSizeBytes(uiMatrixSize*uiMatrixSize * sizeof(std::uint32_t));

    auto pAAcc(alpaka::memory::memAlloc<TAccMemorySpace, std::uint32_t>(uiSizeBytes));
    auto pBAcc(alpaka::memory::memAlloc<TAccMemorySpace, std::uint32_t>(uiSizeBytes));
    auto pCAcc(alpaka::memory::memAlloc<TAccMemorySpace, std::uint32_t>(uiSizeBytes));

    alpaka::memory::memCopy<TAccMemorySpace, alpaka::MemorySpaceHost>(pAAcc.get(), vuiA.data(), uiSizeBytes);
    alpaka::memory::memCopy<TAccMemorySpace, alpaka::MemorySpaceHost>(pBAcc.get(), vuiB.data(), uiSizeBytes);
    alpaka::memory::memCopy<TAccMemorySpace, alpaka::MemorySpaceHost>(pCAcc.get(), vuiC.data(), uiSizeBytes);

    // Build the kernel executor.
    auto exec(alpaka::createKernelExecutor<TAcc, TKernel>());
    // Profile the kernel execution.
    profileAcceleratedKernel(exec(workSize), uiMatrixSize, pAAcc.get(), pBAcc.get(), pCAcc.get());

    // Copy back the result.
    alpaka::memory::memCopy<alpaka::MemorySpaceHost, TAccMemorySpace>(vuiC.data(), pCAcc.get(), uiSizeBytes);

    // Assert that the results are correct. 
    // When multiplying square matrices filled with ones, the result of each cell is the size of the matrix. 
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
        alpaka::logEnabledAccelerators();

        std::cout << std::endl;

        // Select the first cuda device to print out its properties. 
        // NOTE: This is not required to run any kernels on the CUDA accelerator.
#ifdef ALPAKA_CUDA_ENABLED
        alpaka::AccCuda::setDevice(0);
#endif

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