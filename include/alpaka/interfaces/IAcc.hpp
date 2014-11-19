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

#include <alpaka/core/Vec.hpp>          // alpaka::vec
#include <alpaka/core/Positioning.hpp>  // alpaka::origin::Grid/Blocks

#include <boost/mpl/placeholders.hpp>   // boost::mpl::_1

//-----------------------------------------------------------------------------
//! The name space for the accelerator library.
//-----------------------------------------------------------------------------
namespace alpaka
{
    //#############################################################################
    //! The interface for all accelerators.
    //!
    //! All the methods of this interface are declared ALPAKA_FCT_HOST_ACC. 
    //! Because the kernel is always compiled with ALPAKA_FCT_HOST_ACC for all accelerators (even for AccSerial), equivalently there has to be an implementation of all methods for host and device for all accelerators. 
    //! These device functions are not implemented and will not call the underlying implementation for the device code, because this will never be executed and would not compile.
    //
    // TODO: Implement:
    // int __syncthreads_count(int predicate):  evaluates predicate for all threads of the block and returns the number of threads for which predicate evaluates to non-zero.
    // int __syncthreads_and(int predicate):    evaluates predicate for all threads of the block and returns non-zero if and only if predicate evaluates to non-zero for all of them.
    // int __syncthreads_or(int predicate):     evaluates predicate for all threads of the block and returns non-zero if and only if predicate evaluates to non-zero for any of them.
    //
    // __threadfence_block();       wait until memory accesses are visible to block
    // __threadfence();             wait until memory accesses are visible to block and device
    // __threadfence_system();      wait until memory accesses are visible to block and device and host(2.x)
    //
    // __all(predicate):            Evaluate predicate for all active threads of the warp and return non-zero if and only if predicate evaluates to non-zero for all of them.Supported by devices of compute capability 1.2 and higher.
    // __any(predicate):            Evaluate predicate for all active threads of the warp and return non-zero if and only if predicate evaluates to non-zero for any of them.Supported by devices of compute capability 1.2 and higher.
    // __ballot(predicate):         Evaluate predicate for all active threads of the warp and return an integer whose Nth bit is set if and only if predicate evaluates to non-zero for the Nth thread of the warp and the Nth thread is active.Supported by devices of compute capability 2.0 and higher.
    // For each of these warp vote operations, the result excludes threads that are inactive (e.g., due to warp divergence). Inactive threads are represented by 0 bits in the value returned by __ballot() and are not considered in the reductions performed by __all() and __any().
    //
    // rsqrtf
    //
    // sincosf
    //
    // clock_t clock();
    // long long int clock64();
    //
    // The read-only data cache load function is only supported by devices of compute capability 3.5 and higher.
    // T __ldg(const T* address);
    //
    // surface and texture access
    //
    // __shfl, __shfl_up, __shfl_down, __shfl_xor exchange a variable between threads within a warp.

    //#############################################################################
    template<typename TAcc = boost::mpl::_1>
    class IAcc :
        protected TAcc
    {
    public:
        using MemorySpace = typename TAcc::MemorySpace;

    public:
        //-----------------------------------------------------------------------------
        //! \return [The maximum number of memory sharing kernel executions | The maximum block size] allowed by the underlying accelerator.
        // TODO: Check if the used size is valid!
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST static vec<3u> getSizeBlockKernelsMax()
        {
            return TAcc::getSizeBlockKernelsMax();
        }
        //-----------------------------------------------------------------------------
        //! \return [The maximum number of memory sharing kernel executions | The maximum block size] allowed by the underlying accelerator.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST static std::uint32_t getSizeBlockKernelsLinearMax()
        {
            return TAcc::getSizeBlockKernelsLinearMax();
        }

        //-----------------------------------------------------------------------------
        //! \return The requested size.
        //-----------------------------------------------------------------------------
        template<typename TOrigin, typename TUnit, typename TDimensionality = dim::D3>
        ALPAKA_FCT_HOST_ACC typename detail::DimToRetType<TDimensionality>::type getSize() const
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
        ALPAKA_FCT_HOST_ACC typename detail::DimToRetType<TDimensionality>::type getIdx() const
        {
#ifndef __CUDA_ARCH__
            return TAcc::template getIdx<TOrigin, TUnit, TDimensionality>();
#else
            return {};
#endif
        }

        //-----------------------------------------------------------------------------
        //! Execute the atomic operation on the given address with the given value.
        //! \return The old value before executing the atomic operation.
        //-----------------------------------------------------------------------------
        template<typename TOp, typename T>
        ALPAKA_FCT_HOST_ACC T atomicOp(T * const addr, T const & value) const
        {
#ifndef __CUDA_ARCH__
            return TAcc::template atomicOp<TOp, T>(addr, value);
#else
            return {};
#endif
        }

        //-----------------------------------------------------------------------------
        //! Syncs all kernels in the current block.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST_ACC void syncBlockKernels() const
        {
#ifndef __CUDA_ARCH__
            return TAcc::syncBlockKernels();
#endif
        }

        //-----------------------------------------------------------------------------
        //! \return Allocates block shared memory.
        //-----------------------------------------------------------------------------
        template<typename T, std::size_t TuiNumElements>
        ALPAKA_FCT_HOST_ACC T * allocBlockSharedMem() const
        {
            static_assert(TuiNumElements > 0, "The number of elements to allocate in block shared memory must not be zero!");
#ifndef __CUDA_ARCH__
            return TAcc::template allocBlockSharedMem<T, TuiNumElements>();
#else
            return nullptr;
#endif
        }

        //-----------------------------------------------------------------------------
        //! \return The pointer to the externally allocated block shared memory.
        //-----------------------------------------------------------------------------
        template<typename T>
        ALPAKA_FCT_HOST_ACC T * getBlockSharedExternMem() const
        {
#ifndef __CUDA_ARCH__
            return TAcc::template getBlockSharedExternMem<T>();
#else
            return nullptr;
#endif
        }
    };
}
