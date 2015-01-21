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
#include <typeindex>					// std::type_index
#include <unordered_map>				// std::unordered_map

#if BOOST_COMP_MSVC
 #pragma warning(push)
 #pragma warning(disable: 4512)			// boost/program_options/options_description.hpp(265): warning C4512: 'boost::program_options::options_description': assignment operator was implicitly defined as deleted
#endif

#include <boost/program_options.hpp>	// boost::program_options

#if BOOST_COMP_MSVC
 #pragma warning(pop)
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
class MatMulKernel :
	public TAcc
{
	public:

	//-----------------------------------------------------------------------------
	//! The kernel.
	//-----------------------------------------------------------------------------
	template<
        typename TElem>
	ALPAKA_FCT_ACC void operator()(
		std::size_t const n,
		TElem const * const A,
		TElem const * const B,
		TElem * const C) const
	{
		// Column and row of C to calculate.
		auto const v2uiGridKernelIdx(TAcc::template getIdx<alpaka::Grid, alpaka::Kernels>().template subvec<2u>());
		auto const & cx(v2uiGridKernelIdx[0]);
		auto const & cy(v2uiGridKernelIdx[1]);
		// Column and row inside the block of C to calculate.
		auto const v2uiBlockKernelIdx(TAcc::template getIdx<alpaka::Block, alpaka::Kernels>().template subvec<2u>());
		auto const & col(v2uiBlockKernelIdx[0]);
		auto const & row(v2uiBlockKernelIdx[1]);

		auto const v2uiBlockKernelsExtent(TAcc::template getExtent<alpaka::Block,alpaka::Kernels>().template subvec<2u>());
		auto const & uiBlockKernelsExtentX(v2uiBlockKernelsExtent[0]);
		auto const & uiBlockKernelsExtentY(v2uiBlockKernelsExtent[1]);
		auto const uiBlockSizeLin(uiBlockKernelsExtentX * uiBlockKernelsExtentY);

		// Shared memory used to store the current blocks of A and B.
		auto * const pBlockSharedA(TAcc::template getBlockSharedExternMem<TElem>());
		auto * const pBlockSharedB(pBlockSharedA + uiBlockSizeLin);

		TElem fCSum(0);

		// If the element is outside of the matrix, write zero into the shared block and prevent out-of-bounds access to A and B
		bool const bOutsideMatrix((cx >= n) || (cy >= n));

		// Loop over all blocks of A and B that are required to compute the C block.
		auto const uiGridSizeX(TAcc::template getExtent<alpaka::Grid, alpaka::Blocks>()[0]);
		for(std::size_t l(0); l < uiGridSizeX; ++l)
		{
			// Copy data to shared memory.
			auto const uiIndexA(cy * n + l * uiBlockKernelsExtentX + col);
			pBlockSharedA[row * uiBlockKernelsExtentX + col] = bOutsideMatrix
				? 0
				: A[uiIndexA];
			auto const uiIndexB((l * uiBlockKernelsExtentX + row) * n + cx);
			pBlockSharedB[row * uiBlockKernelsExtentX + col] = bOutsideMatrix
				? 0
				: B[uiIndexB];

			// Synchronize to make sure the sub-matrices are loaded before starting the computation.
			TAcc::syncBlockKernels();

			// Dyadic product within shared memory.
			for(std::uint32_t k(0); k < uiBlockKernelsExtentY; ++k)
			{
				fCSum += pBlockSharedA[row * uiBlockKernelsExtentX + k]
					* pBlockSharedB[k * uiBlockKernelsExtentX + col];
			}

			// Synchronize to make sure that the preceding computation is done before loading two new sub-matrices of A and B in the next iteration.
			TAcc::syncBlockKernels();
		}

		if(!bOutsideMatrix)
		{
			C[cy * n + cx] += fCSum;
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
            typename TElem>
		ALPAKA_FCT_HOST static std::size_t getBlockSharedExternMemSizeBytes(
			alpaka::Vec<3u> const & v3uiBlockKernelsExtent,
			std::size_t const,
			TElem const * const,
			TElem const * const,
			TElem * const)
		{
			// Reserve the buffer for the two blocks of A and B.
			return 2u * v3uiBlockKernelsExtent.prod() * sizeof(TElem);
		}
	};
}

//-----------------------------------------------------------------------------
//! Profiles the given kernel.
//-----------------------------------------------------------------------------
template<
    typename TExec,
    typename... TArgs>
void profileAcceleratedKernel(
	TExec const & exec,
	TArgs && ... args)
{
	std::cout
		<< "profileAcceleratedKernel("
		<< " kernelExecutor: " << typeid(TExec).name()
		<< ")" << std::endl;

	auto const tpStart(std::chrono::high_resolution_clock::now());

	// Execute the accelerated kernel.
	exec(std::forward<TArgs>(args) ...);

	// Enqueue an event to wait for. This allows synchronization after the (possibly) asynchronous kernel execution.
	alpaka::event::GetEventT<TExec::Acc> ev;
	alpaka::event::enqueue(ev);
	alpaka::wait::wait(ev);

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

		// Let alpaka calculate a good block and grid sizes given our full problem extent.
		alpaka::WorkExtent const workExtent(alpaka::getValidWorkExtent<TAcc>(
			alpaka::Vec<3u>(uiMatrixSize, uiMatrixSize, 1u),
			bAdaptiveBlockKernelsExtent));

		std::cout
			<< "profileAcceleratedMatMulKernel("
			<< " uiMatrixSize:" << uiMatrixSize
			<< ", accelerator: " << alpaka::acc::getAccName<TAcc>()
			<< ", kernel: " << typeid(Kernel).name()
			<< ", workExtent: " << workExtent
			<< ")" << std::endl;

		alpaka::extent::RuntimeExtents<alpaka::dim::Dim2> extent(
            uiMatrixSize,
            uiMatrixSize
		);

        // Allocate the A and B matrices as st::vectors because this allows them to be filled with uint32_t(1).
        // alpaka::memory::set only supports setting all bytes leading to a value of 16843009 in all elements.
        std::vector<std::uint32_t> vuiA(uiMatrixSize * uiMatrixSize, 1u);
        std::vector<std::uint32_t> vuiB(uiMatrixSize * uiMatrixSize, 1u);
        // Wrap the std::vectors into a memory buffer object.
        // For 1D data this would not be required because alpaka::memory::copy is specialized for std::vector and std::array.
        // For multi dimensional data you could directly create them using alpaka::memory::alloc<Type, MemSpaceHost>, which is not used here.
        // Instead we use MemBufPlainPtrWrapper to wrap the data.
        using MemBufWrapper = alpaka::memory::MemBufPlainPtrWrapper<
            alpaka::memory::MemSpaceHost,
            std::uint32_t,
            alpaka::dim::Dim2>;
        MemBufWrapper memBufAHost(vuiA.data(), extent);
        MemBufWrapper memBufBHost(vuiB.data(), extent);

        // Allocate C and set it to zero.
        auto memBufCHost(alpaka::memory::alloc<std::uint32_t, alpaka::memory::MemSpaceHost>(extent));
        alpaka::memory::set(memBufCHost, 0u, extent);

        // Allocate the buffers on the accelerator.
        using AccMemorySpace = typename alpaka::memory::GetMemSpaceT<TAcc>;
		auto memBufAAcc(alpaka::memory::alloc<std::uint32_t, AccMemorySpace>(extent));
		auto memBufBAcc(alpaka::memory::alloc<std::uint32_t, AccMemorySpace>(extent));
		auto memBufCAcc(alpaka::memory::alloc<std::uint32_t, AccMemorySpace>(extent));

        // Copy Host -> Acc.
		alpaka::memory::copy(memBufAAcc, memBufAHost, extent);
		alpaka::memory::copy(memBufBAcc, memBufBHost, extent);
		alpaka::memory::copy(memBufCAcc, memBufCHost, extent);

		// Build the kernel executor.
		auto exec(alpaka::createKernelExecutor<TAcc, Kernel>());
		// Get a new stream.
		alpaka::stream::GetStreamT<TAcc> stream;
		// Profile the kernel execution.
		profileAcceleratedKernel(exec(workExtent, stream),
			uiMatrixSize,
			alpaka::memory::getNativePtr(memBufAAcc),
			alpaka::memory::getNativePtr(memBufBAcc),
			alpaka::memory::getNativePtr(memBufCAcc));

		// Copy back the result.
		alpaka::memory::copy(memBufCHost, memBufCAcc, extent);

		// Assert that the results are correct.
		// When multiplying square matrices filled with ones, the result of each cell is the size of the matrix.
		std::uint32_t const uiCorrectResult(static_cast<std::uint32_t>(uiMatrixSize));

		bool bResultCorrect(true);
        auto const pHostData(alpaka::memory::getNativePtr(memBufCHost));
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
			return 1;
		}
		else
		{
			std::cout << std::endl;
			std::cout << "################################################################################" << std::endl;
			std::cout << "                              alpaka matmul test                                " << std::endl;
			std::cout << "################################################################################" << std::endl;
			std::cout << std::endl;

			// Logs the enabled accelerators.
			alpaka::acc::writeEnabledAccelerators(std::cout);

			std::cout << std::endl;

			assert(vm.count("adaptiveBlockKernelsExtent") > 0);
			bool const bAdaptiveBlockKernelsExtent(vm["adaptiveBlockKernelsExtent"].as<bool>());
			std::cout << "Adaptive block kernel size:" << bAdaptiveBlockKernelsExtent << std::endl;

#ifdef ALPAKA_CUDA_ENABLED
			// Select the first CUDA device.
			// NOTE: This is not required to run any kernels on the CUDA accelerator because all accelerators have a default device. This only shows the possibility.
			alpaka::device::DeviceManager<alpaka::AccCuda>::setCurrentDevice(
				alpaka::device::DeviceManager<alpaka::AccCuda>::getCurrentDevice());
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
