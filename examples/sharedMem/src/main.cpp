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

#include <chrono>                                   // std::chrono::high_resolution_clock
#include <cassert>                                  // assert
#include <iostream>                                 // std::cout
#include <vector>                                   // std::vector
#include <typeinfo>                                 // typeid
#include <utility>                                  // std::forward

//#############################################################################
//! An accelerated test kernel.
//! Uses atomicOp, syncBlockThreads, getBlockSharedExternMem, getIdx, getWorkDiv and global memory to compute a (useless) result.
//! \tparam TAcc The accelerator environment to be executed on.
//! \tparam TuiNumUselessWork The number of useless calculations done in each kernel execution.
//#############################################################################
template<
    typename TuiNumUselessWork>
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
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        std::uint32_t * const puiBlockRetVals,
        std::uint32_t const uiMult2) const
    -> void
    {
        static_assert(
            alpaka::dim::Dim<TAcc>::value == 1,
            "The SharedMemKernel expects 1-dimensional indices!");

        // The number of threads in this block.
        std::size_t const uiNumKernelsInBlock(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]);

        // Get the extern allocated shared memory.
        std::uint32_t * const pBlockShared(acc.template getBlockSharedExternMem<std::uint32_t>());

        // Get some shared memory (allocate a second buffer directly afterwards to check for some synchronization bugs).
        //std::uint32_t * const pBlockShared1(alpaka::block::shared::allocArr<std::uint32_t, TuiNumUselessWork::value>());
        //std::uint32_t * const pBlockShared2(alpaka::block::shared::allocArr<std::uint32_t, TuiNumUselessWork::value>());

        // Calculate linearized index of the thread in the block.
        std::size_t const uiIdxBlockThreadsLin(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);


        // Fill the shared block with the thread ids [1+X, 2+X, 3+X, ..., #Threads+X].
        std::uint32_t iSum1(static_cast<std::uint32_t>(uiIdxBlockThreadsLin+1));
        for(std::uint32_t i(0); i<TuiNumUselessWork::value; ++i)
        {
            iSum1 += i;
        }
        pBlockShared[uiIdxBlockThreadsLin] = iSum1;


        // Synchronize all threads because now we are writing to the memory again but inverse.
        acc.syncBlockThreads();

        // Do something useless.
        std::uint32_t iSum2(static_cast<std::uint32_t>(uiIdxBlockThreadsLin));
        for(std::uint32_t i(0); i<TuiNumUselessWork::value; ++i)
        {
            iSum2 -= i;
        }
        // Add the inverse so that every cell is filled with [#Threads, #Threads, ..., #Threads].
        pBlockShared[(uiNumKernelsInBlock-1)-uiIdxBlockThreadsLin] += iSum2;


        // Synchronize all threads again.
        acc.syncBlockThreads();

        // Now add up all the cells atomically and write the result to cell 0 of the shared memory.
        if(uiIdxBlockThreadsLin > 0)
        {
            alpaka::atomic::atomicOp<alpaka::atomic::op::Add>(acc, &pBlockShared[0], pBlockShared[uiIdxBlockThreadsLin]);
        }


        acc.syncBlockThreads();

        // Only master writes result to global memory.
        if(uiIdxBlockThreadsLin==0)
        {
            // Calculate linearized block id.
            std::size_t const uiblockIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);

            puiBlockRetVals[uiblockIdx] = pBlockShared[0] * m_uiMult * uiMult2;
        }
    }

public:
    std::uint32_t const m_uiMult;
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
                typename TuiNumUselessWork,
                typename TAcc>
            struct BlockSharedExternMemSizeBytes<
                SharedMemKernel<TuiNumUselessWork>,
                TAcc>
            {
                //-----------------------------------------------------------------------------
                //! \return The size of the shared memory allocated for a block.
                //-----------------------------------------------------------------------------
                template<
                    typename TVec,
                    typename... TArgs>
                ALPAKA_FN_HOST static auto getBlockSharedExternMemSizeBytes(
                    TVec const & vuiBlockThreadsExtents,
                    TArgs && ...)
                -> std::uint32_t
                {
                    return vuiBlockThreadsExtents.prod() * sizeof(std::uint32_t);
                }
            };
        }
    }
}

//#############################################################################
//! Profiles the example kernel and checks the result.
//#############################################################################
template<
    typename TuiNumUselessWork>
struct SharedMemTester
{
    template<
        typename TAcc,
        typename TSize,
        typename TVal>
    auto operator()(
        TSize const uiNumElements,
        TVal const uiMult2)
    -> void
    {
        std::cout << std::endl;
        std::cout << "################################################################################" << std::endl;

        // Create the kernel function object.
        SharedMemKernel<TuiNumUselessWork> kernel(42);

        // Get the host device.
        //auto devHost(alpaka::dev::cpu::getDev());

        // Select a device to execute on.
        alpaka::dev::Dev<TAcc> devAcc(
            alpaka::dev::DevMan<TAcc>::getDevByIdx(0));

        // Get a stream on this device.
        alpaka::stream::Stream<alpaka::dev::Dev<TAcc>> stream(
            alpaka::stream::create(devAcc));

        // Set the grid blocks extent.
        alpaka::workdiv::WorkDivMembers<alpaka::dim::DimInt<1u>, TSize> const workDiv(
            alpaka::workdiv::getValidWorkDiv<TAcc>(
                devAcc,
                uiNumElements,
                false,
                alpaka::workdiv::GridBlockExtentsSubDivRestrictions::Unrestricted));

        std::cout
            << "SharedMemTester("
            << " accelerator: " << alpaka::acc::getAccName<TAcc>()
            << ", kernel: " << typeid(kernel).name()
            << ", workDiv: " << workDiv
            << ")" << std::endl;

        TSize const uiGridBlocksCount(
            alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Blocks>(workDiv)[0u]);
        TSize const uiBlockThreadsCount(
            alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(workDiv)[0u]);

        // An array for the return values calculated by the blocks.
        std::vector<TVal> vuiBlockRetVals(uiGridBlocksCount, static_cast<TVal>(0));

        // Allocate accelerator buffers and copy.
        TSize const uiSizeElements(uiGridBlocksCount);
        auto blockRetValsAcc(alpaka::mem::buf::alloc<TVal, TSize>(devAcc, uiSizeElements));
        alpaka::mem::view::copy(blockRetValsAcc, vuiBlockRetVals, uiSizeElements);

        // Create the executor.
        auto exec(alpaka::exec::create<TAcc>(workDiv, stream));
        // Profile the kernel execution.
        std::cout << "Execution time: "
            << alpaka::examples::measureKernelRunTimeMs(
                exec,
                kernel,
                alpaka::mem::view::getPtrNative(blockRetValsAcc),
                uiMult2)
            << " ms"
            << std::endl;

        // Copy back the result.
        alpaka::mem::view::copy(vuiBlockRetVals, blockRetValsAcc, uiSizeElements);

        // Assert that the results are correct.
        TVal const uiCorrectResult(
            static_cast<TVal>(uiBlockThreadsCount*uiBlockThreadsCount)
            * kernel.m_uiMult
            * uiMult2);

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
auto main()
-> int
{
    try
    {
        std::cout << std::endl;
        std::cout << "################################################################################" << std::endl;
        std::cout << "                            alpaka sharedMem test                               " << std::endl;
        std::cout << "################################################################################" << std::endl;
        std::cout << std::endl;

        // Logs the enabled accelerators.
        alpaka::examples::accs::writeEnabledAccs<alpaka::dim::DimInt<1u>, std::uint32_t>(std::cout);

        std::cout << std::endl;

        using TuiNumUselessWork = std::integral_constant<std::size_t, 100u>;
        std::uint32_t const uiMult2(5u);

        SharedMemTester<TuiNumUselessWork> sharedMemTester;

        // Execute the kernel on all enabled accelerators.
        alpaka::core::forEachType<
            alpaka::examples::accs::EnabledAccs<alpaka::dim::DimInt<1u>, std::uint32_t>>(
                sharedMemTester,
                static_cast<std::uint32_t>(512u),
                uiMult2);

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
