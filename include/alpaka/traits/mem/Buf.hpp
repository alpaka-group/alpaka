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

#pragma once

#include <alpaka/traits/Acc.hpp>        // AccT
#include <alpaka/traits/mem/View.hpp>   // ViewType

#include <alpaka/core/Common.hpp>       // ALPAKA_FCT_HOST

namespace alpaka
{
    namespace traits
    {
        //-----------------------------------------------------------------------------
        //! The memory traits.
        //-----------------------------------------------------------------------------
        namespace mem
        {
            //#############################################################################
            //! The memory buffer base type trait.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev,
                typename TSfinae = void>
            struct BufType;

            //#############################################################################
            //! The memory allocator trait.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev,
                typename TSfinae = void>
            struct Alloc;

            //#############################################################################
            //! The memory mapping trait.
            //#############################################################################
            template<
                typename TBuf,
                typename TDev,
                typename TSfinae = void>
            struct Map;

            //#############################################################################
            //! The memory unmapping trait.
            //#############################################################################
            template<
                typename TBuf,
                typename TDev,
                typename TSfinae = void>
            struct Unmap;
        }
    }
    
    //-----------------------------------------------------------------------------
    //! The memory trait accessors.
    //-----------------------------------------------------------------------------
    namespace mem
    {
        //#############################################################################
        //! The memory buffer base type trait alias template to remove the ::type.
        //#############################################################################
        template<
            typename TElem,
            typename TDim,
            typename TDev>
        using BufT = typename traits::mem::BufType<TElem, TDim, TDev>::type;

        //-----------------------------------------------------------------------------
        //! Allocates memory on the given device.
        //!
        //! \tparam TElem The element type of the returned buffer.
        //! \tparam TExtents The extents of the buffer.
        //! \tparam TDev The type of device the buffer is allocated on.
        //! \param dev The device to allocate the buffer on.
        //! \param extents The extents of the buffer.
        //! \return The newly allocated buffer.
        //-----------------------------------------------------------------------------
        template<
            typename TElem,
            typename TExtents,
            typename TDev>
        ALPAKA_FCT_HOST auto alloc(
            TDev const & dev,
            TExtents const & extents = TExtents())
        -> decltype(
            traits::mem::Alloc<
                TElem,
                dim::DimT<TExtents>,
                TDev>
            ::alloc(
                dev,
                extents))
        {
            return traits::mem::Alloc<
                TElem,
                dim::DimT<TExtents>,
                TDev>
            ::alloc(
                dev,
                extents);
        }
        //-----------------------------------------------------------------------------
        //! Maps the buffer into the memory of the given device.
        //!
        //! \tparam TBuf The buffer type.
        //! \tparam TDev The device type.
        //! \param buf The buffer to map into the device memory.
        //! \param dev The device to map the buffer into.
        //-----------------------------------------------------------------------------
        template<
            typename TBuf,
            typename TDev>
        ALPAKA_FCT_HOST auto map(
            TBuf const & buf,
            TDev const & dev)
        -> void
        {
            return traits::mem::Map<
                TBuf,
                TDev>
            ::map(
                buf,
                dev);
        }
        //-----------------------------------------------------------------------------
        //! Unmaps the buffer from the memory of the given device.
        //!
        //! \tparam TBuf The buffer type.
        //! \tparam TDev The device type.
        //! \param buf The buffer to unmap from the device memory.
        //! \param dev The device to unmap the buffer from.
        //-----------------------------------------------------------------------------
        template<
            typename TBuf,
            typename TDev>
        ALPAKA_FCT_HOST auto unmap(
            TBuf const & buf,
            TDev const & dev)
        -> void
        {
            return traits::mem::Unmap<
                TBuf,
                TDev>
            ::unmap(
                buf,
                dev);
        }
    }
}
