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

#include <acc/Vec.hpp>			// acc::vec<3>
#include <acc/Positioning.hpp>	// acc::origin::Grid/Tiles

//-----------------------------------------------------------------------------
//! The namespace for the accelerator library.
//-----------------------------------------------------------------------------
namespace acc
{
	namespace detail
	{
		//#############################################################################
		//! The interface for all accelerators.
		//!
		//! All the methods of this interface are declared ACC_FCT_CPU_CUDA. 
		//! Because the kernel is always compiled with ACC_FCT_CPU_CUDA for all accelerators (even for AccSerial), equivalently there has to be an implementation of all methods for host and device for all accelerators. 
		//! These device functions are not implemented and will not call the underlying implementation for the device code, because this will never be executed and would not compile.
		//
		// TODO: Implement:
		// __syncthreads_count: Voting of a conditional statement over the block.
		// __threadfence_block(); 	wait until memory accesses are visible to block
		// __threadfence();       	wait until memory accesses are visible to block and device
		// __threadfence_system();	wait until memory accesses are visible to block and device and host(2.x)
		// atomics
		// old = atomicSub(&addr, value);  // old = *addr;  *addr –= value
		// old = atomicExch(&addr, value);  // old = *addr;  *addr  = value
		// old = atomicMin(&addr, value);  // old = *addr;  *addr = min( old, value )
		// old = atomicMax(&addr, value);  // old = *addr;  *addr = max( old, value )
		// increment up to value, then reset to 0  
		// decrement down to 0, then reset to value
		// old = atomicInc(&addr, value);  // old = *addr;  *addr = ((old >= value) ? 0 : old+1 )
		// old = atomicDec(&addr, value);  // old = *addr;  *addr = ((old == 0) or (old > val) ? val : old–1 )
		// old = atomicAnd(&addr, value);  // old = *addr;  *addr &= value
		// old = atomicOr(&addr, value);  // old = *addr;  *addr |= value
		// old = atomicXor(&addr, value);  // old = *addr;  *addr ^= value
		// compare-and-store
		// old = atomicCAS(&addr, compare, value);  // old = *addr;  *addr = ((old == compare) ? value : old)
		//
		// NOTE: CUDA: "__syncthreads() is allowed in conditional code but only if the conditional evaluates identically across the entire thread block, otherwise the code execution is likely to hang or produce unintended side effects."
		// This is not allowed on other accelerators then CUDA.
		//#############################################################################
		template<typename TAcc>
		class IAcc :
			public TAcc
		{
		public:
			//-----------------------------------------------------------------------------
			//! \return [The maximum number of memory sharing kernel executions | The maximum tile size] allowed by the underlying accelerator.
			// TODO: Check if the used size is valid!
			//-----------------------------------------------------------------------------
			ACC_FCT_CPU_CUDA static vec<3> getSizeTileKernelsMax()
			{
				return TAcc::getSizeTileKernelsMax();
			}
			//-----------------------------------------------------------------------------
			//! \return [The maximum number of memory sharing kernel executions | The maximum tile size] allowed by the underlying accelerator.
			//-----------------------------------------------------------------------------
			ACC_FCT_CPU static std::uint32_t getSizeTileKernelsLinearMax()
			{
				return TAcc::getSizeTileKernelsLinearMax();
			}

			//-----------------------------------------------------------------------------
			//! \return The requested size.
			//-----------------------------------------------------------------------------
			template<typename TOrigin, typename TUnit, typename TDimensionality = dim::D3>
			ACC_FCT_CPU_CUDA typename DimToRetType<TDimensionality>::type getSize() const
			{
#ifndef __CUDA_ARCH__
				return TAcc::template getSize<TOrigin, TUnit, TDimensionality>();
#else
				return {};
#endif
			}

		protected:
			//-----------------------------------------------------------------------------
			//! \return The requested index.
			//-----------------------------------------------------------------------------
			template<typename TOrigin, typename TUnit, typename TDimensionality = dim::D3>
			ACC_FCT_CPU_CUDA typename DimToRetType<TDimensionality>::type getIdx() const
			{
#ifndef __CUDA_ARCH__
				return TAcc::template getIdx<TOrigin, TUnit, TDimensionality>();
#else
				return{};
#endif
			}

			//-----------------------------------------------------------------------------
			//! Atomic addition.
			//-----------------------------------------------------------------------------
			template<typename T>
			ACC_FCT_CPU_CUDA void atomicFetchAdd(T * sum, T summand) const
			{
#ifndef __CUDA_ARCH__
				return TAcc::template atomicFetchAdd<T>(sum, summand);
#endif
			}

			//-----------------------------------------------------------------------------
			//! Syncs all threads in the current block.
			//-----------------------------------------------------------------------------
			ACC_FCT_CPU_CUDA void syncTileKernels() const
			{
#ifndef __CUDA_ARCH__
				return TAcc::syncTileKernels();
#endif
			}

			//-----------------------------------------------------------------------------
			//! \return The pointer to the block shared memory.
			//-----------------------------------------------------------------------------
			template<typename T>
			ACC_FCT_CPU_CUDA T * getTileSharedExternMem() const
			{
#ifndef __CUDA_ARCH__
				return TAcc::template getTileSharedExternMem<T>();
#else
				return nullptr;
#endif
			}
		};
	}
}