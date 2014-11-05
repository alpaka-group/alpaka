/**
* Copyright 2014 Benjamin Worpitz
*
* This file is part of acc.
*
* acc is free software: you can redistribute it and/or modify
* it under the terms of of either the GNU General Public License or
* the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* acc is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License and the GNU Lesser General Public License
* for more details.
*
* You should have received a copy of the GNU General Public License
* and the GNU Lesser General Public License along with acc.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include <acc/IAcc.hpp>			// acc::IAcc<...>
#include <acc/Executor.hpp>		// acc::buildKernelExecutor<...>
#include <acc/WorkSize.hpp>		// acc::WorkSizeDefault

#include <chrono>				// std::chrono::high_resolution_clock
#include <cassert>				// assert
#include <iostream>				// std::cout
#include <vector>				// std::vector
#include <typeinfo>				// typeid

#ifdef ACC_CUDA_ENABLED
	#include <cuda.h>
#endif

//#############################################################################
//! An accelerated test kernel.
//! Uses atomicFetchAdd(), syncTileKernels(), shared memory, getIdx, getSize, global memory to compute a (useless) result.
//! \param TAcc The accelerator environment to be executed on.
//
// NOTE: The weird syntax for 'TAcc::template getSize<...>' is required by standard but not all compilers enforce it.
// 'TAcc' is required to make the function call dependent so that its resolution is delayed.
// 'template' is required: http://stackoverflow.com/questions/3786360/confusing-template-error
//#############################################################################
template<typename TAcc>
class ExampleAcceleratedKernel :
	public TAcc
{
public:
	//-----------------------------------------------------------------------------
	//! The kernel.
	//-----------------------------------------------------------------------------
	ACC_FCT_CPU_CUDA void operator()(std::uint32_t * const puiBlockRetVals, std::uint32_t const uiNumUselessWork) const
	{
		//acc::vec<3> const uiIdxTileKernels(TAcc::template getIdx<acc::Tile, acc::Kernels>());
		//acc::vec<3> const v3uiSizeTileKernels(TAcc::template getSize<acc::Tile, acc::Kernels>());
		//acc::vec<3> const v3uiSizeGridTiles(TAcc::template getSize<acc::Grid, acc::Tiles>());

		// The number of threads in this block.
		std::uint32_t const uiNumKernelsInTile(TAcc::template getSize<acc::Tile, acc::Kernels, acc::Linear>());

		// Get the extern allocated shared memory.
		std::uint32_t * const pTileShared(TAcc::template getTileSharedExternMem<std::uint32_t>());

		//std::uint32_t * const pBlockShared1(getTileSharedMem<std::uint32_t, 32>());
		//std::uint32_t * const pBlockShared2(getTileSharedMem<std::uint32_t, 16>());

		// Calculate linearized index of the thread in the block.
		std::uint32_t const uiIdxTileKernelsLin(TAcc::template getIdx<acc::Tile, acc::Kernels, acc::Linear>());

		// Fill the shared block with the thread ids [1+X, 2+X, 3+X, ..., #Threads+X].
		std::uint32_t iSum1(uiIdxTileKernelsLin+1);
		for(std::uint32_t i(0); i<uiNumUselessWork; ++i)
		{
			iSum1 += i;
		}
		pTileShared[uiIdxTileKernelsLin] = iSum1;

		// Synchronize all threads because now we are writing to the memory again but inverse.
		TAcc::syncTileKernels();

		// Do something useless.
		std::uint32_t iSum2(uiIdxTileKernelsLin);
		for(std::uint32_t i(0); i<uiNumUselessWork; ++i)
		{
			iSum2 -= i;
		}

		// Add the inverse so that every cell is filled with [#Threads, #Threads, ..., #Threads].
		pTileShared[(uiNumKernelsInTile-1)-uiIdxTileKernelsLin] += iSum2;

		// Synchronize all threads again.
		TAcc::syncTileKernels();

		// Now add up all the cells atomically and write the result to cell 0 of the shared memory.
		if(uiIdxTileKernelsLin > 0)
		{
			TAcc::atomicFetchAdd(&pTileShared[0], pTileShared[uiIdxTileKernelsLin]);
		}

		TAcc::syncTileKernels();

		// Only master writes result to global mem.
		if(uiIdxTileKernelsLin==0)
		{
			//acc::vec<3> const blockIdx(TAcc::getIdxGridTile());
			// Calculate linearized block id.
			std::uint32_t const bId(TAcc::template getIdx<acc::Grid, acc::Tiles, acc::Linear>());

			puiBlockRetVals[bId] = pTileShared[0];
		}
	}

	//-----------------------------------------------------------------------------
	//! \return The size of the shared memory allocated for a block.
	//-----------------------------------------------------------------------------
	static std::size_t getBlockSharedMemSizeBytes(acc::vec<3> const v3uiSizeTileKernels)
	{
		return v3uiSizeTileKernels.prod() * sizeof(std::uint32_t);
	}
};

//-----------------------------------------------------------------------------
//! Profiles the given kernel.
//-----------------------------------------------------------------------------
template<typename TAcc, template<typename> class TKernel, typename TWorkSize, typename... TArgs>
void profileAcceleratedKernel(std::size_t const uiIterations, TWorkSize const & workSize, TArgs && ... args)
{
	std::cout
		<< "profileAcceleratedKernel("
		//<< " kernel: " << typeid(TKernel).name()
		<< "accelerator: " << typeid(TAcc).name()
		<< ", iterations: " << uiIterations
		<< ", workSize: " << workSize << ")"
		<< std::endl;

	auto const tpStart(std::chrono::high_resolution_clock::now());

	for(std::size_t i(0); i<uiIterations; ++i)
	{
		// Execute the accelerated kernel.
		acc::buildKernelExecutor<TAcc, TKernel>(workSize)(std::forward<TArgs>(args)...);
	}

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
	void operator()(std::size_t const uiIterations, TWorkSize const & workSize, std::uint32_t * const puiBlockRetVals, std::uint32_t const uiNumUselessWork)
	{
		profileAcceleratedKernel<TAcc, ExampleAcceleratedKernel>(uiIterations, workSize, puiBlockRetVals, uiNumUselessWork);
	}
};

#ifdef ACC_CUDA_ENABLED
//-----------------------------------------------------------------------------
//! Profiles the example kernel (specialized for CUDA).
//-----------------------------------------------------------------------------
template<>
class AcceleratedExampleKernelProfiler<acc::AccCuda>
{
public:
	template<typename TWorkSize>
	void operator()(std::size_t const uiIterations, TWorkSize const & workSize, std::uint32_t * const puiBlockRetVals, std::uint32_t const uiNumUselessWork)
	{
		std::size_t const uiNumTilesInGrid(workSize.template getSize<acc::Grid, acc::Tiles, acc::Linear>());

		std::uint32_t * pBlockRetValsDev (nullptr);
		std::size_t const uiNumBytes(uiNumTilesInGrid * sizeof(std::uint32_t));
		ACC_CUDA_CHECK(cudaMalloc((void **) &pBlockRetValsDev, uiNumBytes));
		ACC_CUDA_CHECK(cudaMemcpy(pBlockRetValsDev, puiBlockRetVals, uiNumBytes, cudaMemcpyHostToDevice));

		profileAcceleratedKernel<acc::AccCuda, ExampleAcceleratedKernel>(uiIterations, workSize, pBlockRetValsDev, uiNumUselessWork);
		ACC_CUDA_CHECK(cudaDeviceSynchronize());

		ACC_CUDA_CHECK(cudaMemcpy(puiBlockRetVals, pBlockRetValsDev, uiNumBytes, cudaMemcpyDeviceToHost));
		ACC_CUDA_CHECK(cudaFree(pBlockRetValsDev)); 
	}
};
#endif

//-----------------------------------------------------------------------------
//! Profiles the example kernel and checks the result.
//-----------------------------------------------------------------------------
template<typename TAcc, typename TWorkSize>
void profileAcceleratedExampleKernel(std::size_t const uiIterations, TWorkSize const & workSize, std::uint32_t const uiNumUselessWork)
{
	std::size_t const uiNumTilesInGrid(workSize.template getSize<acc::Grid, acc::Tiles, acc::Linear>());
	std::size_t const uiNumKernelsInTile(workSize.template getSize<acc::Tile, acc::Kernels, acc::Linear>());

	// An array for the return values calculated by the blocks.
	std::vector<std::uint32_t> vuiBlockRetVals(uiNumTilesInGrid, 0);

	AcceleratedExampleKernelProfiler<TAcc>()(uiIterations, workSize, vuiBlockRetVals.data(), uiNumUselessWork);

	// Assert that the results are correct.
	bool bResultCorrect(true);
	for(std::size_t i(0); i<uiNumTilesInGrid; ++i)
	{
		if(static_cast<std::size_t>(vuiBlockRetVals[i]) != uiNumKernelsInTile*uiNumKernelsInTile)
		{
			std::cout << "vuiBlockRetVals[" << i << "] == " << vuiBlockRetVals[i] << " != " << uiNumKernelsInTile*uiNumKernelsInTile << std::endl;
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
#ifdef ACC_SERIAL_ENABLED
	std::cout << "ACC_SERIAL_ENABLED ";
#endif
#ifdef ACC_THREADS_ENABLED
	std::cout << "ACC_THREADS_ENABLED ";
#endif
#ifdef ACC_FIBERS_ENABLED
	std::cout << "ACC_FIBERS_ENABLED ";
#endif
#ifdef ACC_OPENMP_ENABLED
	std::cout << "ACC_OPENMP_ENABLED ";
#endif
#ifdef ACC_CUDA_ENABLED
	std::cout << "ACC_CUDA_ENABLED ";
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

#ifdef ACC_CUDA_ENABLED
	acc::AccCuda::setDevice(0);
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
		std::cout << "                              acc basic test                                    " << std::endl;
		std::cout << "################################################################################" << std::endl;
		std::cout << std::endl;

		// Logs the enabled accelerators.
		logEnabledAccelerators();

		std::cout << std::endl;
		// Initialize the accelerators.
		initAccelerators();

		// Set the grid size.
		acc::vec<3> const v3uiSizeGridTiles(16u, 8u, 4u);

		// Set the block size (to the minimum all enabled tests support).
		acc::vec<3> v3uiSizeTileKernels(
#if defined ACC_SERIAL_ENABLED
		1u, 1u, 1u
#elif defined ACC_OPENMP_ENABLED
		4u, 4u, 2u
#elif defined ACC_CUDA_ENABLED || defined ACC_THREADS_ENABLED || defined ACC_FIBERS_ENABLED
		16u, 16u, 2u
#else
		1u, 1u, 1u
#endif
		);

		std::uint32_t const uiNumUselessWork(1000);

		acc::WorkSize const workSize(v3uiSizeGridTiles, v3uiSizeTileKernels);

		std::size_t const uiIterations(1);

#ifdef ACC_SERIAL_ENABLED
		std::cout << std::endl;
		std::cout << "################################################################################" << std::endl;
		profileAcceleratedExampleKernel<acc::AccSerial>(uiIterations, workSize, uiNumUselessWork);
		std::cout << "################################################################################" << std::endl;
#endif
#ifdef ACC_THREADS_ENABLED
		std::cout << std::endl;
		std::cout << "################################################################################" << std::endl;
		profileAcceleratedExampleKernel<acc::AccThreads>(uiIterations, workSize, uiNumUselessWork);
		std::cout << "################################################################################" << std::endl;
#endif
#ifdef ACC_FIBERS_ENABLED
		std::cout << std::endl;
		std::cout << "################################################################################" << std::endl;
		profileAcceleratedExampleKernel<acc::AccFibers>(uiIterations, workSize, uiNumUselessWork);
		std::cout << "################################################################################" << std::endl;
#endif
#ifdef ACC_OPENMP_ENABLED
		std::cout << std::endl;
		std::cout << "################################################################################" << std::endl;
		profileAcceleratedExampleKernel<acc::AccOpenMp>(uiIterations, workSize, uiNumUselessWork);
		std::cout << "################################################################################" << std::endl;
#endif
#ifdef ACC_CUDA_ENABLED
		std::cout << std::endl;
		std::cout << "################################################################################" << std::endl;
		profileAcceleratedExampleKernel<acc::AccCuda>(uiIterations, workSize, uiNumUselessWork);
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