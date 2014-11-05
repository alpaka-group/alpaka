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
#include <acc/Index.hpp>					// IIndex

#include <cstddef>							// std::size_t
#include <cstdint>							// unit8_t
#include <vector>							// std::vector
#include <thread>							// std::thread
#include <map>								// std::map
#include <algorithm>						// std::for_each
#include <mutex>							// std::mutex
#include <condition_variable>				// std::condition_variable
#include <array>							// std::array
#include <cassert>							// assert
#include <stdexcept>						// std::except
#include <string>							// std::to_string
#include <functional>						// std::bind
#ifdef _DEBUG
	#include <iostream>						// std::cout
#endif

namespace acc
{
	//#############################################################################
	//! This class stores the current indices as members.
	//#############################################################################
	class IndexThreads
	{
	public:
		//-----------------------------------------------------------------------------
		//! Default-constructor.
		//-----------------------------------------------------------------------------
		ACC_FCT_CPU_CUDA IndexThreads() = default;

		//-----------------------------------------------------------------------------
		//! Copy-onstructor.
		//-----------------------------------------------------------------------------
		ACC_FCT_CPU_CUDA IndexThreads(IndexThreads const & other) = default;

		//-----------------------------------------------------------------------------
		//! \return The thread index of the currently executed kernel.
		//-----------------------------------------------------------------------------
		ACC_FCT_CPU_CUDA vec<3> getIdxTileKernel() const
		{
			auto const idThread(std::this_thread::get_id());
			auto const itFind(m_mThreadsToIndices.find(idThread));
			assert(itFind != m_mThreadsToIndices.end());

			return itFind->second;
		}
		//-----------------------------------------------------------------------------
		//! \return The block index of the currently executed kernel.
		//-----------------------------------------------------------------------------
		ACC_FCT_CPU_CUDA vec<3> getIdxGridTile() const
		{
			return m_v3uiGridTileIdx;
		}
	protected:
		mutable std::map<
			std::thread::id,
			vec<3>> m_mThreadsToIndices;						//!< The mapping of thread id's to thread indices.
		mutable vec<3> m_v3uiGridTileIdx;						//!< The index of the currently executed block.
	};

	//#############################################################################
	//! A barrier.
	//#############################################################################
	class ThreadBarrier
	{
	public:
		//-----------------------------------------------------------------------------
		//! Constructor.
		//-----------------------------------------------------------------------------
		ACC_FCT_CPU explicit ThreadBarrier(std::size_t const uiNumThreadsToWaitFor = 0) :
			m_uiNumThreadsToWaitFor{uiNumThreadsToWaitFor}
		{}

		//-----------------------------------------------------------------------------
		//! Copy-constructor.
		//-----------------------------------------------------------------------------
		ACC_FCT_CPU ThreadBarrier(ThreadBarrier const & other) :
			m_uiNumThreadsToWaitFor(other.m_uiNumThreadsToWaitFor)
		{}
		//-----------------------------------------------------------------------------
		//! Assignment-operator.
		//-----------------------------------------------------------------------------
		ACC_FCT_CPU ThreadBarrier & operator=(ThreadBarrier const &) = delete;

		//-----------------------------------------------------------------------------
		//! Waits for all the other threads to reach the barrier.
		//-----------------------------------------------------------------------------
		ACC_FCT_CPU void wait()
		{
			std::unique_lock<std::mutex> lock(m_mtxBarrier);
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
		ACC_FCT_CPU void reset(std::size_t const uiNumThreadsToWaitFor)
		{
			std::lock_guard<std::mutex> lock(m_mtxBarrier);
			m_uiNumThreadsToWaitFor = uiNumThreadsToWaitFor;
		}

	private:
		std::mutex m_mtxBarrier;
		std::condition_variable m_cvAllThreadsReachedBarrier;
		std::size_t m_uiNumThreadsToWaitFor;
	};

	//#############################################################################
	//! The base class for all C++11 std::thread accelerated kernels.
	//#############################################################################
	class AccThreads :
		protected detail::IIndex<detail::IndexThreads>,
		public detail::IWorkSize<detail::WorkSizeDefault>
	{
		using TIndex = detail::IIndex<detail::IndexThreads>;
		using TWorkSize = detail::IWorkSize<detail::WorkSizeDefault>;
	public:
		//-----------------------------------------------------------------------------
		//! Constructor.
		//-----------------------------------------------------------------------------
		ACC_FCT_CPU AccThreads() = default;

		//-----------------------------------------------------------------------------
		//! Copy-constructor.
		// Has to be explicitly defined because 'std::mutex::mutex(const std::mutex&)' is deleted.
		// Do not copy most members because they are initialized by the executor for each accelerated execution.
		//-----------------------------------------------------------------------------
		ACC_FCT_CPU AccThreads(AccThreads const & other) :
			TWorkSize(other),
			m_vThreadsInBlock(),
			m_mThreadsToIndices(),
			m_v3uiGridTileIdx(),
			m_mThreadsToBarrier(),
			m_mtxBarrier(),
			m_abarSyncThreads(),
			m_vuiSharedMem(),
			m_mtxAtomicAdd()
		{}

		//-----------------------------------------------------------------------------
		//! Move-constructor.
		// Has to be explicitly defined because 'std::mutex::mutex(const std::mutex&)' is deleted.
		// Do not copy anything because all the members are initialized by the executor for each accelerated execution.
		//-----------------------------------------------------------------------------
		/*ACC_FCT_CPU AccThreads(AccThreads && other) :
			TWorkSize(other),
			m_vThreadsInBlock(std::move(other.m_vThreadsInBlock)),
			m_mThreadsToIndices(std::move(other.m_mThreadsToIndices)),
			m_v3uiGridTileIdx(std::move(other.m_v3uiGridTileIdx)),
			m_mThreadsToBarrier(std::move(other.m_mThreadsToBarrier)),
			m_mtxBarrier(),
			m_abarSyncThreads(),
			m_vuiSharedMem(std::move(other.m_vuiSharedMem)),
			m_mtxAtomicAdd()
		{}*/

		//-----------------------------------------------------------------------------
		//! \return The maximum number of kernels in each dimension of a tile allowed.
		//-----------------------------------------------------------------------------
		ACC_FCT_CPU static vec<3> getSizeTileKernelsMax()
		{
			auto const uiSizeTileKernelsLinearMax(getSizeTileKernelsLinearMax());
			return {uiSizeTileKernelsLinearMax, uiSizeTileKernelsLinearMax, uiSizeTileKernelsLinearMax};
		}
		//-----------------------------------------------------------------------------
		//! \return The maximum number of kernels in a tile allowed by.
		//-----------------------------------------------------------------------------
		ACC_FCT_CPU static std::uint32_t getSizeTileKernelsLinearMax()
		{
			// FIXME: What is the maximum? Just set a reasonable value? There is a implementation defined maximum where the creation of a new thread crashes.
			// std::thread::hardware_concurrency is too small but a multiple of it? But it can return 0, so a default for this case?
			return 1024;	// Magic number.
		}

	protected:
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
			// TODO: We could use a list of mutexes and lock the mutex depending on the target variable to allow multiple atomicFetchAdd`s on different targets concurrently.
			std::lock_guard<std::mutex> lock(m_mtxAtomicAdd);
			auto & rsum = *sum;
			rsum += summand;
		}
		//-----------------------------------------------------------------------------
		//! Syncs all threads in the current block.
		//-----------------------------------------------------------------------------
		ACC_FCT_CPU void syncTileKernels() const
		{
			auto const idThread(std::this_thread::get_id());
			auto const itFind(m_mThreadsToBarrier.find(idThread));
			assert(itFind != m_mThreadsToBarrier.end());

			auto & uiBarIndex(itFind->second);
			std::size_t const uiBarrierIndex(uiBarIndex % 2);

			auto & bar(m_abarSyncThreads[uiBarrierIndex]);

			// (Re)initialize a barrier if this is the first thread to reach it.
			if(bar.getNumThreadsToWaitFor() == 0)
			{
				std::lock_guard<std::mutex> lock(m_mtxBarrier);
				if(bar.getNumThreadsToWaitFor() == 0)
				{
					bar.reset(m_uiNumKernelsPerTile);
				}
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
		mutable std::vector<std::thread> m_vThreadsInBlock;		//!< The threads executing the current block.

		// syncTileKernels
		mutable std::size_t m_uiNumKernelsPerTile;				//!< The number of kernels per tile the barrier has to wait for.
		mutable std::map<
			std::thread::id,
			std::size_t> m_mThreadsToBarrier;					//!< The mapping of thread id's to their current barrier.
		mutable std::mutex m_mtxBarrier;
		mutable ThreadBarrier m_abarSyncThreads[2];				//!< The barriers for the synchronzation of threads. 
																//!< We have to keep the current and the last barrier because one of the threads can reach the next barrier before a other thread was wakeup from the last one and has checked if it can run.
		// getBlockSharedEcternMem
		mutable std::vector<uint8_t> m_vuiSharedMem;			//!< Block shared memory.

		// atomicFetchAdd
		mutable std::mutex m_mtxAtomicAdd;						//!< The mutex protecting access for a atomic operation.
																//!< TODO: This is very slow! Is there a better way? Currently not only the access to the same variable is protected by a mutex but all atomicFetchAdd`s on all threads.

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
				(*static_cast<typename AccThreads::TWorkSize*>(this)) = workSize;
#ifdef _DEBUG
				std::cout << "AccThreads::KernelExecutor()" << std::endl;
#endif
			}

			//-----------------------------------------------------------------------------
			//! Executes the accelerated kernel.
			//-----------------------------------------------------------------------------
			template<typename... TArgs>
			void operator()(TArgs && ... args) const
			{
#ifdef _DEBUG
				std::cout << "[+] AccThreads::KernelExecutor::operator()" << std::endl;
#endif
				auto const uiNumKernelsPerTile(this->AccThreads::template getSize<Tile, Kernels, Linear>());
				auto const uiMaxKernelsPerTile(AccThreads::getSizeTileKernelsLinearMax());
				if(uiNumKernelsPerTile > uiMaxKernelsPerTile)
				{
					throw std::runtime_error(("The given tileSize '" + std::to_string(uiNumKernelsPerTile) + "' is larger then the supported maximum of '" + std::to_string(uiMaxKernelsPerTile) + "' by the threads accelerator!").c_str());
				}

				auto const v3uiSizeGridTiles(this->AccThreads::template getSize<Grid, Tiles, D3>());
				auto const v3uiSizeTileKernels(this->AccThreads::template getSize<Tile, Kernels, D3>());
#ifdef _DEBUG
				//std::cout << "GridTiles: " << v3uiSizeGridTiles << " TileKernels: " << v3uiSizeTileKernels << std::endl;
#endif

				this->AccThreads::m_uiNumKernelsPerTile = uiNumKernelsPerTile;

				this->AccThreads::m_vuiSharedMem.resize(TAccedKernel::getBlockSharedMemSizeBytes(v3uiSizeTileKernels));

				// CUDA programming guide: "Thread blocks are required to execute independently: It must be possible to execute them in any order, in parallel or in series. 
				// This independence requirement allows thread blocks to be scheduled in any order across any number of cores"
				// -> We can execute them serially.
				for(std::uint32_t bz(0); bz<v3uiSizeGridTiles[2]; ++bz)
				{
					this->AccThreads::m_v3uiGridTileIdx[2] = bz;
					for(std::uint32_t by(0); by<v3uiSizeGridTiles[1]; ++by)
					{
						this->AccThreads::m_v3uiGridTileIdx[1] = by;
						for(std::uint32_t bx(0); bx<v3uiSizeGridTiles[0]; ++bx)
						{
							this->AccThreads::m_v3uiGridTileIdx[0] = bx;

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
										auto threadKernelFct([this](vec<3> const v3uiTileKernelIdx, TArgs ... args){threadKernel<TArgs...>(v3uiTileKernelIdx, args...); });
										this->AccThreads::m_vThreadsInBlock.push_back(std::thread(threadKernelFct, v3uiTileKernelIdx, args...));
#else
										this->AccThreads::m_vThreadsInBlock.push_back(std::thread(&KernelExecutor::threadKernel<TArgs...>, this, v3uiTileKernelIdx, args...));
#endif
									}
								}
							}
							// Join all the threads.
							std::for_each(this->AccThreads::m_vThreadsInBlock.begin(), this->AccThreads::m_vThreadsInBlock.end(),
								[](std::thread & t)
								{
									t.join();
								}
							);
							// Clean up.
							this->AccThreads::m_vThreadsInBlock.clear();
							this->AccThreads::m_mThreadsToIndices.clear();
							this->AccThreads::m_mThreadsToBarrier.clear();

							// After a block has been processed, the shared memory can be deleted.
							this->AccThreads::m_vuiSharedMem.clear();
						}
					}
				}
#ifdef _DEBUG
				std::cout << "[-] AccThreads::KernelExecutor::operator()" << std::endl;
#endif
			}
		private:
			//-----------------------------------------------------------------------------
			//! The thread entry point.
			//-----------------------------------------------------------------------------
			template<typename... TArgs>
			void threadKernel(vec<3> const v3uiTileKernelIdx, TArgs ... args) const
			{
				// We have to store the thread data before the kernel is calling any of the methods of this class depending on them.
				auto const idThread(std::this_thread::get_id());

				{
					// Save the thread id, and index.
#ifdef _MSC_VER	// GCC <= 4.7.2 is not standard conformant and has no member emplace. This works with 4.7.3+.
					this->AccThreads::m_mThreadsToIndices.emplace(idThread, v3uiTileKernelIdx);
					this->AccThreads::m_mThreadsToBarrier.emplace(idThread, 0);
#else
					this->AccThreads::m_mThreadsToIndices.insert(std::make_pair(idThread, v3uiTileKernelIdx));
					this->AccThreads::m_mThreadsToBarrier.insert(std::make_pair(idThread, 0));
#endif
				}

				// Sync all threads so that the maps with thread id's are complete and not changed after here.
				this->AccThreads::syncTileKernels();

				// Execute the kernel itself.
				this->TAccedKernel::operator()(args ...);

				// We have to sync all threads here because if a thread would finish before all threads have been started, the new thread could get a recyled (then duplicate) thread id!
				this->AccThreads::syncTileKernels();
			}
		};
	};

	namespace detail
	{
		//#############################################################################
		//! The threads kernel executor builder.
		//#############################################################################
		template<template<typename> class TKernel, typename TWorkSize>
		class KernelExecutorBuilder<AccThreads, TKernel, TWorkSize>
		{
		public:
			using TAccedKernel = TKernel<IAcc<AccThreads>>;
			using TKernelExecutor = AccThreads::KernelExecutor<TAccedKernel>;

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