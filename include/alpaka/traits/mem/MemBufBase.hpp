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

#include <alpaka/core/Common.hpp>           // ALPAKA_FCT_HOST

#include <alpaka/traits/Dim.hpp>            // DimT
#include <alpaka/traits/mem/MemSpace.hpp>   // MemSpaceT
#include <alpaka/traits/mem/MemBuf.hpp>     // MemAlloc

#include <cstdint>                          // std::uint8_t

namespace alpaka
{
    namespace traits
    {
        namespace mem
        {
            //#############################################################################
            //! The memory buffer base trait.
            //#############################################################################
            template<
                typename TMemBuf, 
                typename TSfinae = void>
            struct IsMemBufBase
            {
                static const bool value = false;
            };

            //#############################################################################
            //! The memory allocator trait.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim, 
                typename TMemSpace, 
                typename TSfinae = void>
            struct MemAlloc;

            //#############################################################################
            //! The memory set trait.
            //!
            //! Fills the buffer with data.
            //#############################################################################
            template<
                typename TDim, 
                typename TMemSpace, 
                typename TSfinae = void>
            struct MemSet;
        }
    }
    
    namespace mem
    {
        //#############################################################################
        //! The memory buffer base trait.
        //#############################################################################
        template<
            typename TMemBuf, 
            typename TSfinae = void>
        using IsMemBufBase = traits::mem::IsMemBufBase<TMemBuf>;

        //-----------------------------------------------------------------------------
        //! Allocates memory in the given memory space.
        //!
        //! \tparam T The type of the returned buffer.
        //! \tparam TMemSpace The memory space to allocate in.
        //! \param extents The extents of the buffer.
        //! \return Pointer to newly allocated buffer.
        //-----------------------------------------------------------------------------
        template<
            typename TElem, 
            typename TMemSpace, 
            typename TExtents>
        ALPAKA_FCT_HOST auto alloc(
            TExtents const & extents = TExtents())
            -> decltype(traits::mem::MemAlloc<TElem, dim::DimT<TExtents>, MemSpaceT<TMemSpace>>::memAlloc(std::declval<TExtents>()))
        {
            return traits::mem::MemAlloc<TElem, dim::DimT<TExtents>, MemSpaceT<TMemSpace>>::memAlloc(
                extents);
        }

        //-----------------------------------------------------------------------------
        //! Sets the memory to the given value.
        //!
        //! \param memBuf The memory buffer to fill.
        //! \param byte Value to set for each element of the specified buffer.
        //! \param extents The extents of the buffer to fill.
        //-----------------------------------------------------------------------------
        template<
            typename TMemBufBase, 
            typename TExtents>
        ALPAKA_FCT_HOST void set(
            TMemBufBase & memBuf, 
            std::uint8_t const & byte, 
            TExtents const & extents)
        {
            static_assert(
                mem::IsMemBufBase<TMemBufBase>::value,
                "The buffer has to be a base buffer!");
            static_assert(
                std::is_same<dim::DimT<TMemBufBase>, dim::DimT<TExtents>>::value,
                "The buffer and the extents are required to have the same dimensionality!");

            traits::mem::MemSet<dim::DimT<TMemBufBase>, MemSpaceT<TMemBufBase>>::memSet(
                memBuf,
                byte,
                extents);
        }

        //-----------------------------------------------------------------------------
        //! Sets the memory to the given value asynchronously.
        //!
        //! \param memBuf The memory buffer to fill.
        //! \param byte Value to set for each element of the specified buffer.
        //! \param extents The extents of the buffer to fill.
        //-----------------------------------------------------------------------------
        template<
            typename TMemBufBase, 
            typename TExtents,
            typename TStream>
        ALPAKA_FCT_HOST void set(
            TMemBufBase & memBuf, 
            std::uint8_t const & byte, 
            TExtents const & extents,
            TStream const & stream)
        {
            static_assert(
                mem::IsMemBufBase<TMemBufBase>::value,
                "The buffer has to be a base buffer!");
            static_assert(
                std::is_same<dim::DimT<TMemBufBase>, dim::DimT<TExtents>>::value,
                "The buffer and the extents are required to have the same dimensionality!");

            traits::mem::MemSet<dim::DimT<TMemBufBase>, MemSpaceT<TMemBufBase>, TStream>::memSetAsync(
                memBuf,
                byte,
                extents,
                stream);
        }
    }
}
