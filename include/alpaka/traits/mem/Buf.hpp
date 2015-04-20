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
        namespace mem
        {
            //#############################################################################
            //! The memory allocator trait.
            //#############################################################################
            template<
                typename TDev,
                typename TElem,
                typename TDim,
                typename TSpace,
                typename TSfinae = void>
            struct Alloc;
        }
    }

    namespace mem
    {
        //-----------------------------------------------------------------------------
        //! Allocates memory in the given memory space.
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
        -> decltype(traits::mem::Alloc<TDev, TElem, dim::DimT<TExtents>, SpaceT<acc::AccT<TDev>>>::alloc(std::declval<TDev const &>(), std::declval<TExtents const &>()))
        {
            return traits::mem::Alloc<
                TDev,
                TElem,
                dim::DimT<TExtents>,
                SpaceT<acc::AccT<TDev>>>
            ::alloc(
                dev,
                extents);
        }
    }
}
