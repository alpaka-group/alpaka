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
#include <vector>							// std::vector
#include <cassert>							// assert
#include <stdexcept>						// std::except
#include <utility>							// std::forward
#include <string>							// std::to_string
#ifdef _DEBUG
	#include <iostream>						// std::cout
#endif

namespace acc
{
	namespace detail
	{
		//#############################################################################
		//! This class stores the current indices as members.
		//#############################################################################
		class IndexDefault
		{
		public:
			//-----------------------------------------------------------------------------
			//! Default-constructor.
			//-----------------------------------------------------------------------------
			ACC_FCT_CPU_CUDA IndexDefault() = default;

			//-----------------------------------------------------------------------------
			//! Copy-onstructor.
			//-----------------------------------------------------------------------------
			ACC_FCT_CPU_CUDA IndexDefault(IndexDefault const & other) = default;

			//-----------------------------------------------------------------------------
			//! \return The thread index of the currently executed kernel.
			//-----------------------------------------------------------------------------
			ACC_FCT_CPU_CUDA vec<3> getIdxTileKernel() const
			{
				return m_v3uiTileKernelIdx;
			}
			//-----------------------------------------------------------------------------
			//! \return The block index of the currently executed kernel.
			//-----------------------------------------------------------------------------
			ACC_FCT_CPU_CUDA vec<3> getIdxGridTile() const
			{
				return m_v3uiGridTileIdx;
			}

		private:
			vec<3> m_v3uiTileKernelIdx;
			vec<3> m_v3uiGridTileIdx;
		};
	}

	//#############################################################################
	//! The base class for all non accelerated kernels.
	//#############################################################################
	class AccSerial :
		protected detail::IIndex<detail::IndexDefault>,
		public detail::IWorkSize<detail::WorkSizeDefault>
	{
		using TIndex = detail::IIndex<detail::IndexDefault>;
		using TWorkSize = detail::IWorkSize<detail::WorkSizeDefault>;
	public:
		//-----------------------------------------------------------------------------
		//! Constructor.
		//-----------------------------------------------------------------------------
		ACC_FCT_CPU AccSerial() = default;

		//-----------------------------------------------------------------------------
		//! \return The maximum number of kernels in each dimension of a tile allowed.
		//-----------------------------------------------------------------------------
		ACC_FCT_CPU static vec<3> getSizeTileKernelsMax()
		{
			auto const uiSizeTileKernelsLinearMax(getSizeTileKernelsLinearMax());
			return {uiSizeTileKernelsLinearMax, uiSizeTileKernelsLinearMax, uiSizeTileKernelsLinearMax};
		}
		//-----------------------------------------------------------------------------
		//! \return The maximum number of kernels in a tile allowed.
		//-----------------------------------------------------------------------------
		ACC_FCT_CPU static std::uint32_t getSizeTileKernelsLinearMax()
		{
			return 1;
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
			auto & rsum = *sum;
			rsum += summand;
		}

		//-----------------------------------------------------------------------------
		//! Syncs all threads in the current block.
		//-----------------------------------------------------------------------------
		ACC_FCT_CPU void syncTileKernels() const
		{
			// Nothing to do in here because only one thread in a group is allowed.
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
		mutable std::vector<uint8_t> m_vuiSharedMem;	//!< Block shared memory.

		mutable vec<3> m_v3uiTileKernelIdx;			//!< The index of the currently executed kernel.
		mutable vec<3> m_v3uiGridTileIdx;				//!< The index of the currently executed block.

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
				(*static_cast<typename AccSerial::TWorkSize*>(this)) = workSize;
#ifdef _DEBUG
				std::cout << "AccSerial::KernelExecutor()" << std::endl;
#endif
			}

			//-----------------------------------------------------------------------------
			//! Executes the accelerated kernel.
			//-----------------------------------------------------------------------------
			template<typename... TArgs>
			void operator()(TArgs && ... args) const
			{
#ifdef _DEBUG
				std::cout << "[+] AccSerial::KernelExecutor::operator()" << std::endl;
#endif
				auto const uiNumKernelsPerTile(this->AccSerial::template getSize<Tile, Kernels, Linear>());
				auto const uiMaxKernelsPerTile(AccSerial::getSizeTileKernelsLinearMax());
				if(uiNumKernelsPerTile > uiMaxKernelsPerTile)
				{
					throw std::runtime_error(("The given tileSize '" + std::to_string(uiNumKernelsPerTile) + "' is larger then the supported maximum of '" + std::to_string(uiMaxKernelsPerTile) + "' by the serial accelerator!").c_str());
				}

				auto const v3uiSizeGridTiles(this->AccSerial::template getSize<Grid, Tiles, D3>());
				auto const v3uiSizeTileKernels(this->AccSerial::template getSize<Tile, Kernels, D3>());
#ifdef _DEBUG
				//std::cout << "GridTiles: " << v3uiSizeGridTiles << " TileKernels: " << v3uiSizeTileKernels << std::endl;
#endif

				this->AccSerial::m_vuiSharedMem.resize(TAccedKernel::getBlockSharedMemSizeBytes(v3uiSizeTileKernels));

				for(std::uint32_t bz(0); bz<v3uiSizeGridTiles[2]; ++bz)
				{
					this->AccSerial::m_v3uiGridTileIdx[2] = bz;
					for(std::uint32_t by(0); by<v3uiSizeGridTiles[1]; ++by)
					{
						this->AccSerial::m_v3uiGridTileIdx[1] = by;
						for(std::uint32_t bx(0); bx<v3uiSizeGridTiles[0]; ++bx)
						{
							this->AccSerial::m_v3uiGridTileIdx[0] = bx;

							for(std::uint32_t tz(0); tz<v3uiSizeTileKernels[2]; ++tz)
							{
								this->AccSerial::m_v3uiTileKernelIdx[2] = tz;
								for(std::uint32_t ty(0); ty<v3uiSizeTileKernels[1]; ++ty)
								{
									this->AccSerial::m_v3uiTileKernelIdx[1] = ty;
									for(std::uint32_t tx(0); tx<v3uiSizeTileKernels[0]; ++tx)
									{
										this->AccSerial::m_v3uiTileKernelIdx[0] = tx;

										this->TAccedKernel::operator()(std::forward<TArgs>(args)...);
									}
								}
							}
						}
					}
				}

				// After all blocks have been processed, the shared memory can be deleted.
				this->AccSerial::m_vuiSharedMem.clear();
#ifdef _DEBUG
				std::cout << "[-] AccSerial::KernelExecutor::operator()" << std::endl;
#endif
			}
		};
	};

	namespace detail
	{
		//#############################################################################
		//! The serial kernel executor builder.
		//#############################################################################
		template<template<typename> class TKernel, typename TWorkSize>
		class KernelExecutorBuilder<AccSerial, TKernel, TWorkSize>
		{
		public:
			using TAccedKernel = TKernel<IAcc<AccSerial>>;
			using TKernelExecutor = AccSerial::KernelExecutor<TAccedKernel>;

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