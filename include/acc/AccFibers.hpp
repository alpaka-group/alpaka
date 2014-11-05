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

#pragma once

#include <acc/KernelExecutorBuilder.hpp>	// KernelExecutorBuilder
#include <acc/WorkSize.hpp>					// IWorkSize, WorkSizeDefault

#include <cstddef>							// std::size_t
#include <cstdint>							// unit8_t
#include <cassert>							// assert
#include <stdexcept>						// std::except

// Force the usage of variadics for fibers.
#define BOOST_FIBERS_USE_VARIADIC_FIBER

// Boost fiber: 
// http://olk.github.io/libs/fiber/doc/html/index.html
// https://github.com/olk/boost-fiber
#include <boost/fiber/fiber.hpp>			// boost::fibers::fiber
#include <boost/fiber/operations.hpp>		// boost::this_fiber
#include <boost/fiber/condition.hpp>		// boost::fibers::condition_variable
#include <boost/fiber/mutex.hpp>			// boost::fibers::mutex
//#include <boost/fiber/barrier.hpp>		// boost::fibers::barrier

namespace acc
{
	//#############################################################################
	//! A barrier.
	// NOTE: We do not use the boost::fibers::barrier because it does not support simple resetting.
	//#############################################################################
	class FiberBarrier
	{
	public:
		//-----------------------------------------------------------------------------
		//! Constructor.
		//-----------------------------------------------------------------------------
		ACC_FCT_CPU explicit FiberBarrier(std::size_t uiNumThreadsToWaitFor = 0) :
			m_uiNumThreadsToWaitFor{uiNumThreadsToWaitFor}
		{
		}

		//-----------------------------------------------------------------------------
		//! Deleted copy-constructor.
		//-----------------------------------------------------------------------------
		ACC_FCT_CPU FiberBarrier(FiberBarrier const &) = delete;
		//-----------------------------------------------------------------------------
		//! Deleted assignment-operator.
		//-----------------------------------------------------------------------------
		ACC_FCT_CPU FiberBarrier & operator=(FiberBarrier const &) = delete;

		//-----------------------------------------------------------------------------
		//! Waits for all the other threads to reach the barrier.
		//-----------------------------------------------------------------------------
		ACC_FCT_CPU void wait()
		{
			boost::unique_lock<boost::fibers::mutex> lock(m_mtxBarrier);
			if(--m_uiNumThreadsToWaitFor == 0)
			{
				m_cvAllThreadsReachedBarrier.notify_all();
			}
			else
			{
				m_cvAllThreadsReachedBarrier.wait(lock, [this] { return m_uiNumThreadsToWaitFor == 0; });
			}
		}

		//-----------------------------------------------------------------------------
		//! \return The number of threads to wait for.
		//! NOTE: The value almost always is invalid the time you get it.
		//-----------------------------------------------------------------------------
		ACC_FCT_CPU std::size_t getNumThreadsToWaitFor() const
		{
			return m_uiNumThreadsToWaitFor;
		}

		//-----------------------------------------------------------------------------
		//! Resets the number of threads to wait for to the given number.
		//-----------------------------------------------------------------------------
		ACC_FCT_CPU void reset(std::size_t uiNumThreadsToWaitFor)
		{
			//boost::unique_lock<boost::fibers::mutex> lock(m_mtxBarrier);
			m_uiNumThreadsToWaitFor = uiNumThreadsToWaitFor;
		}

	private:
		boost::fibers::mutex m_mtxBarrier;
		boost::fibers::condition_variable m_cvAllThreadsReachedBarrier;
		std::size_t m_uiNumThreadsToWaitFor;
	};

	//#############################################################################
	//! The base class for all C++11 std::thread accelerated kernels.
	//#############################################################################
	template<typename... TArgs>
	class AccFibers :
		public detail::IWorkSize<detail::WorkSizeDefault>
	{
		using TWorkSize = detail::IWorkSize<detail::WorkSizeDefault>;
	public:
		//-----------------------------------------------------------------------------
		//! Constructor.
		//-----------------------------------------------------------------------------
		ACC_FCT_CPU AccFibers() = default;

		//-----------------------------------------------------------------------------
		//! Deleted copy-constructor.
		//-----------------------------------------------------------------------------
		ACC_FCT_CPU AccFibers(AccFibers const &) = delete;
		//-----------------------------------------------------------------------------
		//! Deleted assignment-operator.
		//-----------------------------------------------------------------------------
		ACC_FCT_CPU AccFibers & operator=(AccFibers const &) = delete;

		//-----------------------------------------------------------------------------
		//! \return The maximum number of kernels in each dimension of a tile allowed.
		//-----------------------------------------------------------------------------
		ACC_FCT_CPU static vec<3> getSizeTileKernelsMax()
		{
			auto const uiSizeTileKernelsLinearMax(getSizeTileKernelsLinearMax());
			return {uiSizeTileKernelsLinearMax, uiSizeTileKernelsLinearMax, uiSizeTileKernelsLinearMax};
		}
		//-----------------------------------------------------------------------------
		//! \return The maximum number of kernels in a tile allowed by the underlying accelerator.
		//-----------------------------------------------------------------------------
		ACC_FCT_CPU static std::uint32_t getSizeTileKernelsLinearMax()
		{
			// FIXME: What is the maximum? Just set a reasonable value?
			return 1024;	// Magic number.
		}

	protected:
		//-----------------------------------------------------------------------------
		//! \return The thread index of the currently executed kernel.
		//-----------------------------------------------------------------------------
		ACC_FCT_CPU vec<3> getIdxTileKernel() const
		{
			auto const idFiber(boost::this_fiber::get_id());
			auto const itFind(m_mFibersToIndices.find(idFiber));
			assert(itFind != m_mFibersToIndices.end());

			return itFind->second;
		}
		//-----------------------------------------------------------------------------
		//! \return The block index of the currently executed kernel.
		//-----------------------------------------------------------------------------
		ACC_FCT_CPU vec<3> getIdxGridTile() const
		{
			return m_v3uiGridTileIdx;
		}

		//-----------------------------------------------------------------------------
		//! \return The requested index.
		//-----------------------------------------------------------------------------
		template<typename TOrigin, typename TUnit, typename TDimensionality = dim::D3>
		ACC_FCT_CPU_CUDA typename DimToRetType<TDimensionality>::type getIdx() const
		{
			return TIndex::getIdx<TOrigin, TUnit, TDimensionality>(*static_cast<TWorkSize * const>(this));
		}

		//-----------------------------------------------------------------------------
		//! Atomic addition.
		//-----------------------------------------------------------------------------
		template<typename T>
		ACC_FCT_CPU void atomicFetchAdd(T * sum, T summand) const
		{
			// No mutex required because fibers are thread sequential.
			auto & rsum = *sum;
			rsum += summand;
		}
		//-----------------------------------------------------------------------------
		//! Syncs all threads in the current block.
		//-----------------------------------------------------------------------------
		ACC_FCT_CPU void syncTileKernels() const
		{
			auto const idFiber(boost::this_fiber::get_id());
			auto const itFind(m_mFibersToBarrier.find(idFiber));
			assert(itFind != m_mFibersToBarrier.end());

			auto & uiBarIndex(itFind->second);
			std::size_t const uiBarrierIndex(uiBarIndex % 2);

			auto & bar(m_abarSyncFibers[uiBarrierIndex]);

			// (Re)initialize a barrier if this is the first thread to reach it.
			if(bar.getNumThreadsToWaitFor() == 0)
			{
				bar.reset(m_uiNumKernelsPerTile);
			}

			// Wait for the barrier.
			bar.wait();
			++uiBarIndex;
		}

		//-----------------------------------------------------------------------------
		//! \return The pointer to the block shared memory.
		//-----------------------------------------------------------------------------
		template<typename T>
		ACC_FCT_CPU T * getTileSharedExternMem() const
		{
			return reinterpret_cast<T*>(m_vuiSharedMem.data());
		}

	private:
		mutable std::vector<boost::fibers::fiber> m_vFibersInBlock;	//!< The fibers executing the current block.

		// getXxxIdx
		mutable std::map<
			boost::fibers::fiber::id,
			vec<3>> m_mFibersToIndices;								//!< The mapping of fibers id's to fibers indices.
		mutable vec<3> m_v3uiGridTileIdx;							//!< The index of the currently executed block.

		// syncTileKernels
		mutable std::size_t m_uiNumKernelsPerTile;					//!< The number of kernels per tile the barrier has to wait for.
		mutable std::map<
			boost::fibers::fiber::id,
			std::size_t> m_mFibersToBarrier;						//!< The mapping of fibers id's to their current barrier.
		mutable FiberBarrier m_abarSyncFibers[2];					//!< The barriers for the synchronzation of fibers. 
																	//!< We have the keep to current and the last barrier because one of the fibers can reach the next barrier before a other fiber was wakeup from the last one and has checked if it can run.
		// getTileSharedExternMem
		mutable boost::fibers::fiber::id m_idMasterFiber;			//!< The id of the master fiber.
		mutable std::vector<uint8_t> m_vuiSharedMem;				//!< Block shared memory.

	public:
		//#############################################################################
		//! The executor for an accelerated serial kernel.
		//#############################################################################
		template<typename TAccedKernel>
		class KernelExecutor :
			protected TAccedKernel
		{
		public:
			//-----------------------------------------------------------------------------
			//! Constructor.
			//-----------------------------------------------------------------------------
			template<typename TWorkSize>
			KernelExecutor(TWorkSize workSize)
			{
				(*static_cast<typename AccFibers::TWorkSize*>(this)) = workSize;
#ifdef _DEBUG
				std::cout << "AccFibers::KernelExecutor()" << std::endl;
#endif
			}

			//-----------------------------------------------------------------------------
			//! Executes the accelerated kernel.
			//-----------------------------------------------------------------------------
			template<typename... TArgs>
			void operator()(TArgs && ... args) const
			{
#ifdef _DEBUG
				std::cout << "[+] AccFibers::KernelExecutor::operator()" << std::endl;
#endif
				auto const uiNumKernelsPerTile(this->AccFibers::template getSize<Tile, Kernels, Linear>());
				auto const uiMaxKernelsPerTile(AccFibers::getSizeTileKernelsLinearMax());
				if(uiNumKernelsPerTile > uiMaxKernelsPerTile)
				{
					throw std::runtime_error(("The given tileSize '" + std::to_string(uiNumKernelsPerTile) + "' is larger then the supported maximum of '" + std::to_string(uiMaxKernelsPerTile) + "' by the fibers accelerator!").c_str());
				}

				auto const v3uiSizeGridTiles(this->AccFibers::template getSize<Grid, Tiles, D3>());
				auto const v3uiSizeTileKernels(this->AccFibers::template getSize<Tile, Kernels, D3>());
#ifdef _DEBUG
				//std::cout << "GridTiles: " << v3uiSizeGridTiles << " TileKernels: " << v3uiSizeTileKernels << std::endl;
#endif

				this->AccFibers::m_uiNumKernelsPerTile = uiNumKernelsPerTile;

				this->AccFibers::m_vuiSharedMem.resize(TAccedKernel::getBlockSharedMemSizeBytes(v3uiSizeTileKernels));

				// CUDA programming guide: "Thread blocks are required to execute independently: It must be possible to execute them in any order, in parallel or in series. 
				// This independence requirement allows thread blocks to be scheduled in any order across any number of cores"
				// -> We can execute them serially.
				for(std::uint32_t bz(0); bz<v3uiSizeGridTiles[2]; ++bz)
				{
					this->AccFibers::m_v3uiGridTileIdx[2] = bz;
					for(std::uint32_t by(0); by<v3uiSizeGridTiles[1]; ++by)
					{
						this->AccFibers::m_v3uiGridTileIdx[1] = by;
						for(std::uint32_t bx(0); bx<v3uiSizeGridTiles[0]; ++bx)
						{
							this->AccFibers::m_v3uiGridTileIdx[0] = bx;

							//std::size_t const uiNumKernelsInTile(getSizeTileKernels()[0] * getSizeTileKernels()[1] * getSizeTileKernels()[2]);

							// This is called automatically if required.
							//m_abarSyncThreads[0].reset(uiNumKernelsInTile);
							//m_abarSyncThreads[1].reset(uiNumKernelsInTile);

							vec<3> v3uiTileKernelIdx;
							for(std::uint32_t tz(0); tz<v3uiSizeTileKernels[2]; ++tz)
							{
								v3uiTileKernelIdx[2] = tz;
								for(std::uint32_t ty(0); ty<v3uiSizeTileKernels[1]; ++ty)
								{
									v3uiTileKernelIdx[1] = ty;
									for(std::uint32_t tx(0); tx<v3uiSizeTileKernels[0]; ++tx)
									{
										v3uiTileKernelIdx[0] = tx;

										// Create the thread.
										// The v3uiTileKernelIdx is required to be copied in from the environment because if the thread is immediately suspended the variable is already changed for the next iteration/thread.
#ifdef _MSC_VER	// MSVC <= 14 do not compile the std::thread constructor because the type of the member function template is missing the this pointer as first argument.
										auto threadKernelFct([this](vec<3> const v3uiTileKernelIdx, TArgs ... args){fiberKernel<TArgs...>(v3uiTileKernelIdx, args...); });
										this->AccFibers::m_vFibersInBlock.push_back(boost::fibers::fiber(threadKernelFct, v3uiTileKernelIdx, args...));
#else
										this->AccFibers::m_vFibersInBlock.push_back(boost::fibers::fiber(&KernelExecutor::fiberKernel<TArgs...>, this, v3uiTileKernelIdx, args...));
#endif
									}
								}
							}
							// Join all the fibers.
							std::for_each(this->AccFibers::m_vFibersInBlock.begin(), this->AccFibers::m_vFibersInBlock.end(),
								[](boost::fibers::fiber & f)
								{
									f.join();
								}
							);
							// Clean up.
							this->AccFibers::m_vFibersInBlock.clear();
							this->AccFibers::m_mFibersToIndices.clear();
							this->AccFibers::m_mFibersToBarrier.clear();

							// After a block has been processed, the shared memory can be deleted.
							this->AccFibers::m_vuiSharedMem.clear();
						}
					}
				}
#ifdef _DEBUG
				std::cout << "[-] AccFibers::KernelExecutor::operator()" << std::endl;
#endif
			}
		private:
			//-----------------------------------------------------------------------------
			//! The thread entry point.
			//-----------------------------------------------------------------------------
			void fiberKernel(vec<3> const v3uiTileKernelIdx, TArgs ... args) const
			{
				// We have to store the thread data before the kernel is calling any of the methods of this class depending on them.
				auto const idFiber(boost::this_fiber::get_id());

				{
					// Save the thread id, and index.
#ifdef _MSC_VER	// GCC <= 4.7.2 is not standard conformant and has no member emplace. This works with 4.7.3+.
					this->AccFibers::m_mFibersToIndices.emplace(idFiber, v3uiTileKernelIdx);
					this->AccFibers::m_mFibersToBarrier.emplace(idFiber, 0);
#else
					this->AccFibers::m_mFibersToIndices.insert(std::pair<boost::fibers::fiber::id, vec<3>>(idFiber, v3uiTileKernelIdx));
					this->AccFibers::m_mFibersToBarrier.insert(std::pair<boost::fibers::fiber::id, vec<3>>(idFiber, 0));
#endif
				}

				// Sync all fibers so that the maps with thread id's are complete and not changed after here.
				this->AccFibers::syncTileKernels();

				// Execute the kernel itself.
				this->TAccedKernel::operator()(args ...);

				// We have to sync all fibers here because if a fiber would finish before all fibers have been started, the new thread could get a recyled (then duplicate) thread id!
				this->AccFibers::syncTileKernels();
			}
		};
	};

	namespace detail
	{
		//#############################################################################
		//! The fibers kernel executor builder.
		//#############################################################################
		template<template<typename> class TKernel, typename TWorkSize>
		class KernelExecutorBuilder<AccFibers, TKernel, TWorkSize>
		{
		public:
			using TAccedKernel = TKernel<IAcc<AccFibers>>;
			using TKernelExecutor = AccFibers::KernelExecutor<TAccedKernel>;

			//-----------------------------------------------------------------------------
			//! Creates an kernel executor for the serial accelerator.
			//-----------------------------------------------------------------------------
			TKernelExecutor operator()(TWorkSize workSize) const
			{
				return TKernelExecutor(workSize);
			}
		};
	}
}