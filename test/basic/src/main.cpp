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
//! An accelerated test kernel.
//! Uses atomicOp(), syncBlockKernels(), shared memory, getIdx, getExtent, global memory to compute a (useless) result.
//! \tparam TAcc The accelerator environment to be executed on.
//! \tparam TuiNumUselessWork The number of useless calculations done in each kernel execution.
//#############################################################################
template<typename TuiNumUselessWork, typename TAcc = alpaka::IAcc<>>
class ExampleAcceleratedKernel :
    public TAcc
{
public:
    //-----------------------------------------------------------------------------
    //! Constructor.
    //-----------------------------------------------------------------------------
    ExampleAcceleratedKernel(std::uint32_t const uiMult = 2) :
        m_uiMult(uiMult)
    {}

    //-----------------------------------------------------------------------------
    //! The kernel.
    //-----------------------------------------------------------------------------
    ALPAKA_FCT_HOST_ACC void operator()(std::uint32_t * const puiBlockRetVals, std::uint32_t const uiMult2) const
    {
        // The number of kernels in this block.
        std::uint32_t const uiNumKernelsInBlock(TAcc::template getExtent<alpaka::Block, alpaka::Kernels, alpaka::Linear>());

        // Get the extern allocated shared memory.
        std::uint32_t * const pBlockShared(TAcc::template getBlockSharedExternMem<std::uint32_t>());

        // Get some shared memory (allocate a second buffer directly afterwards to check for some synchronization bugs).
        //std::uint32_t * const pBlockShared1(TAcc::template allocBlockSharedMem<std::uint32_t, TuiNumUselessWork::value>());
        //std::uint32_t * const pBlockShared2(TAcc::template allocBlockSharedMem<std::uint32_t, TuiNumUselessWork::value>());

        // Calculate linearized index of the kernel in the block.
        std::uint32_t const uiIdxBlockKernelsLin(TAcc::template getIdx<alpaka::Block, alpaka::Kernels, alpaka::Linear>());


        // Fill the shared block with the kernel ids [1+X, 2+X, 3+X, ..., #Threads+X].
        std::uint32_t iSum1(uiIdxBlockKernelsLin+1);
        for(std::uint32_t i(0); i<TuiNumUselessWork::value; ++i)
        {
            iSum1 += i;
        }
        pBlockShared[uiIdxBlockKernelsLin] = iSum1;


        // Synchronize all kernels because now we are writing to the memory again but inverse.
        TAcc::syncBlockKernels();

        // Do something useless.
        std::uint32_t iSum2(uiIdxBlockKernelsLin);
        for(std::uint32_t i(0); i<TuiNumUselessWork::value; ++i)
        {
            iSum2 -= i;
        }
        // Add the inverse so that every cell is filled with [#Kernels, #Kernels, ..., #Kernels].
        pBlockShared[(uiNumKernelsInBlock-1)-uiIdxBlockKernelsLin] += iSum2;


        // Synchronize all kernels again.
        TAcc::syncBlockKernels();

        // Now add up all the cells atomically and write the result to cell 0 of the shared memory.
        if(uiIdxBlockKernelsLin > 0)
        {
            TAcc::template atomicOp<alpaka::Add>(&pBlockShared[0], pBlockShared[uiIdxBlockKernelsLin]);
        }


        TAcc::syncBlockKernels();

        // Only master writes result to global memory.
        if(uiIdxBlockKernelsLin==0)
        {
            // Calculate linearized block id.
            std::uint32_t const bId(TAcc::template getIdx<alpaka::Grid, alpaka::Blocks, alpaka::Linear>());

            puiBlockRetVals[bId] = pBlockShared[0] * m_uiMult * uiMult2;
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
    template<class TuiNumUselessWork, class TAcc>
    struct BlockSharedExternMemSizeBytes<ExampleAcceleratedKernel<TuiNumUselessWork, TAcc>>
    {
        //-----------------------------------------------------------------------------
        //! \return The size of the shared memory allocated for a block.
        //-----------------------------------------------------------------------------
        template<typename... TArgs>
        static std::size_t getBlockSharedExternMemSizeBytes(alpaka::vec<3u> const & v3uiSizeBlockKernels, TArgs && ...)
        {
            return v3uiSizeBlockKernels.prod() * sizeof(std::uint32_t);
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
    alpaka::event::enqueue(ev);
    alpaka::event::wait(ev);

    auto const tpEnd(std::chrono::high_resolution_clock::now());

    auto const durElapsed(tpEnd - tpStart);

    std::cout << "Execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(durElapsed).count() << " ms" << std::endl;
}

//-----------------------------------------------------------------------------
//! Profiles the example kernel and checks the result.
//-----------------------------------------------------------------------------
template<typename TuiNumUselessWork>
struct profileAcceleratedExampleKernel
{
	template<typename TAcc, typename TWorkExtent>
	void operator()(TAcc, alpaka::IWorkExtent<TWorkExtent> const & workExtent, std::uint32_t const uiMult2)
	{
		std::cout << std::endl;
		std::cout << "################################################################################" << std::endl;
		
		using TKernel = ExampleAcceleratedKernel<TuiNumUselessWork>;
		using TAccMemorySpace = typename TAcc::MemorySpace;

		std::cout
			<< "AcceleratedExampleKernelProfiler("
			<< " accelerator: " << typeid(TAcc).name()
			<< ", kernel: " << typeid(TKernel).name()
			<< ", workExtent: " << workExtent
			<< ")" << std::endl;

		std::size_t const uiGridBlocksCount(workExtent.template getExtent<alpaka::Grid, alpaka::Blocks, alpaka::Linear>());
		std::size_t const uiBlockKernelsCount(workExtent.template getExtent<alpaka::Block, alpaka::Kernels, alpaka::Linear>());

		// An array for the return values calculated by the blocks.
		std::vector<std::uint32_t> vuiBlockRetVals(uiGridBlocksCount, 0);

		// Allocate accelerator buffers and copy.
		std::size_t const uiSizeBytes(uiGridBlocksCount * sizeof(std::uint32_t));
		auto pBlockRetValsAcc(alpaka::memory::alloc<TAccMemorySpace, std::uint32_t>(uiSizeBytes));
		alpaka::memory::copy<TAccMemorySpace, alpaka::MemorySpaceHost>(pBlockRetValsAcc.get(), vuiBlockRetVals.data(), uiSizeBytes);

		std::uint32_t const m_uiMult(42);

		auto exec(alpaka::createKernelExecutor<TAcc, TKernel>(m_uiMult));
		profileAcceleratedKernel(exec(workExtent), pBlockRetValsAcc.get(), uiMult2);

		// Copy back the result.
		alpaka::memory::copy<alpaka::MemorySpaceHost, TAccMemorySpace>(vuiBlockRetVals.data(), pBlockRetValsAcc.get(), uiSizeBytes);

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
	}
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
        std::cout << "                              alpaka basic test                                 " << std::endl;
        std::cout << "################################################################################" << std::endl;
        std::cout << std::endl;

        // Logs the enabled accelerators.
        alpaka::logEnabledAccelerators();

        std::cout << std::endl;
		
#ifdef ALPAKA_CUDA_ENABLED
        // Select the first CUDA device. 
        // NOTE: This is not required to run any kernels on the CUDA accelerator because all accelerators have a default device. This only shows the possibility.
        alpaka::device::DeviceManager<alpaka::AccCuda>::setCurrentDevice(
			alpaka::device::DeviceManager<alpaka::AccCuda>::getCurrentDevice());
#endif

        // Set the grid blocks extent.
        alpaka::WorkExtent const workExtent(alpaka::getValidWorkExtent<TAcc>(16u, 8u, 4u), 
            false));

        using TuiNumUselessWork = boost::mpl::int_<100u>;
        std::uint32_t const uiMult2(5u);

		// Execute the kernel on all enabled accelerators.
		boost::mpl::for_each<alpaka::EnabledAccelerators>(
			std::bind(ProfileAcceleratedExampleKernel<TuiNumUselessWork>(), std::placeholders::_1, workExtent, uiMult2)
		);

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