/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of either the GNU General Public License or
* the GNU Lesser General Public License for more details. as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
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
    //! The accelerator interface.
    //
    // All the methods of this interface are declared ALPAKA_FCT_HOST_ACC. 
    // Because the kernel is always compiled with ALPAKA_FCT_HOST_ACC for all accelerators (even for AccSerial), equivalently there has to be an implementation of all methods for host and device for all accelerators. 
    // These device functions are not implemented and will not call the underlying implementation for the device code, because this will never be executed and would not compile.
    //#############################################################################
    template<typename TAcc = boost::mpl::_1>
    class IAcc :
        protected TAcc
    {
    public:
        //-----------------------------------------------------------------------------
        //! \return The requested extent.
        //-----------------------------------------------------------------------------
        template<typename TOrigin, typename TUnit, typename TDimensionality = dim::D3>
        ALPAKA_FCT_HOST_ACC typename detail::DimToRetType<TDimensionality>::type getExtent() const
        {
#ifndef __CUDA_ARCH__
            return TAcc::template getExtent<TOrigin, TUnit, TDimensionality>();
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
