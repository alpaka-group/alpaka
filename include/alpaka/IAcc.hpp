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

#pragma once

#include <alpaka/Vec.hpp>               // alpaka::vec
#include <alpaka/Positioning.hpp>       // alpaka::origin::Grid/Blocks

#include <boost/mpl/placeholders.hpp>   // boost::mpl::_1

//-----------------------------------------------------------------------------
//! The name space for the accelerator library.
//-----------------------------------------------------------------------------
namespace alpaka
{
    //#############################################################################
    //! The interface for all accelerators.
    //!
    //! All the methods of this interface are declared ALPAKA_FCT_CPU_CUDA. 
    //! Because the kernel is always compiled with ALPAKA_FCT_CPU_CUDA for all accelerators (even for AccSerial), equivalently there has to be an implementation of all methods for host and device for all accelerators. 
    //! These device functions are not implemented and will not call the underlying implementation for the device code, because this will never be executed and would not compile.
    //
    // TODO: Implement:
    // __syncthreads_count: Voting of a conditional statement over the block.
    // __threadfence_block();     wait until memory accesses are visible to block
    // __threadfence();           wait until memory accesses are visible to block and device
    // __threadfence_system();    wait until memory accesses are visible to block and device and host(2.x)
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
    template<typename TAcc = boost::mpl::_1>
    class IAcc :
        protected TAcc
    {
    public:
        //-----------------------------------------------------------------------------
        //! \return [The maximum number of memory sharing kernel executions | The maximum block size] allowed by the underlying accelerator.
        // TODO: Check if the used size is valid!
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_CPU_CUDA static vec<3u> getSizeBlockKernelsMax()
        {
            return TAcc::getSizeBlockKernelsMax();
        }
        //-----------------------------------------------------------------------------
        //! \return [The maximum number of memory sharing kernel executions | The maximum block size] allowed by the underlying accelerator.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_CPU static std::uint32_t getSizeBlockKernelsLinearMax()
        {
            return TAcc::getSizeBlockKernelsLinearMax();
        }

        //-----------------------------------------------------------------------------
        //! \return The requested size.
        //-----------------------------------------------------------------------------
        template<typename TOrigin, typename TUnit, typename TDimensionality = dim::D3>
        ALPAKA_FCT_CPU_CUDA typename detail::DimToRetType<TDimensionality>::type getSize() const
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
        ALPAKA_FCT_CPU_CUDA typename detail::DimToRetType<TDimensionality>::type getIdx() const
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
        ALPAKA_FCT_CPU_CUDA void atomicFetchAdd(T * sum, T summand) const
        {
#ifndef __CUDA_ARCH__
            return TAcc::template atomicFetchAdd<T>(sum, summand);
#endif
        }

        //-----------------------------------------------------------------------------
        //! Syncs all kernels in the current block.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_CPU_CUDA void syncBlockKernels() const
        {
#ifndef __CUDA_ARCH__
            return TAcc::syncBlockKernels();
#endif
        }

        //-----------------------------------------------------------------------------
        //! \return Allocates block shared memory.
        //-----------------------------------------------------------------------------
        template<typename T, std::size_t UiNumElements>
        ALPAKA_FCT_CPU_CUDA T * allocBlockSharedMem() const
        {
            static_assert(UiNumElements > 0, "The number of elements to allocate in block shared memory must not be zero!");
#ifndef __CUDA_ARCH__
            return TAcc::template allocBlockSharedMem<T, UiNumElements>();
#else
            return nullptr;
#endif
        }

        //-----------------------------------------------------------------------------
        //! \return The pointer to the externally allocated block shared memory.
        //-----------------------------------------------------------------------------
        template<typename T>
        ALPAKA_FCT_CPU_CUDA T * getBlockSharedExternMem() const
        {
#ifndef __CUDA_ARCH__
            return TAcc::template getBlockSharedExternMem<T>();
#else
            return nullptr;
#endif
        }
    };

	//#############################################################################
	//! The trait for getting the size of the block shared extern memory for a kernel.
	//#############################################################################
    template<typename TAccelereatedKernel>
	struct BlockSharedExternMemSizeBytes
	{
		//-----------------------------------------------------------------------------
		//! \return The size of the shared memory allocated for a block.
		//-----------------------------------------------------------------------------
		static std::size_t getBlockSharedExternMemSizeBytes(vec<3u> const & v3uiSizeBlockKernels)
		{
			return 0;
		}
	};
}