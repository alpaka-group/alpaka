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

#include <alpaka/core/Common.hpp>       // ALPAKA_FCT_HOST

#include <alpaka/traits/Dim.hpp>        // GetDimT
#include <alpaka/traits/Extents.hpp>    // traits::getXXX

#include <type_traits>                  // std::enable_if, std::is_array, std::extent
#include <vector>                       // std::vector
#include <array>                        // std::array

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
            //! The memory space trait.
            //#############################################################################
            template<
                typename T,
                typename TSfinae = void>
            struct GetMemSpace;

            //#############################################################################
            //! The memory element type trait.
            //#############################################################################
            template<
                typename TMemBuf, 
                typename TSfinae = void>
            struct GetMemElem;

            //#############################################################################
            //! The memory buffer type trait.
            //#############################################################################
            template<
                typename TMemSpace,
                typename TElem,
                typename TDim,
                typename TSfinae = void>
            struct GetMemBuf;

            //#############################################################################
            //! The native pointer get trait.
            //#############################################################################
            template<
                typename TMemBuf, 
                typename TSfinae = void>
            struct GetNativePtr;

            //#############################################################################
            //! The pitch in bytes. This is the distance between two consecutive rows.
            //#############################################################################
            template<
                typename TMemBuf,
                typename TSfinae = void>
            struct GetPitchBytes;

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
            //! The memory copy trait.
            //!
            //! Copies memory from one buffer into another buffer possibly in a different memory space.
            //#############################################################################
            template<
                typename TDim, 
                typename TMemSpaceDst, 
                typename TMemSpaceSrc, 
                typename TSfinae = void>
            struct MemCopy;

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

    //-----------------------------------------------------------------------------
    //! The memory trait accessors.
    //-----------------------------------------------------------------------------
    namespace mem
    {
        //#############################################################################
        //! The memory space trait alias template to remove the ::type.
        //#############################################################################
        template<
            typename T>
        using GetMemSpaceT = typename traits::mem::GetMemSpace<T>::type;

        //#############################################################################
        //! The memory element type trait alias template to remove the ::type.
        //#############################################################################
        template<
            typename TMemBuf>
        using GetMemElemT = typename traits::mem::GetMemElem<TMemBuf>::type;

        //#############################################################################
        //! The memory buffer type trait alias template to remove the ::type.
        //#############################################################################
        template<
            typename TElem,
            typename TDim,
            typename TMemSpace>
        using GetMemBufT = typename traits::mem::GetMemBuf<TElem, TDim, GetMemSpaceT<TMemSpace>>::type;

        //-----------------------------------------------------------------------------
        //! Gets the native pointer of the memory buffer.
        //!
        //! \param memBuf The memory buffer to fill.
        //! \return The native pointer.
        //-----------------------------------------------------------------------------
        template<
            typename TMemBuf>
        ALPAKA_FCT_HOST auto getNativePtr(
            TMemBuf const & memBuf)
            -> GetMemElemT<TMemBuf> const *
        {
            return traits::mem::GetNativePtr<TMemBuf>::getNativePtr(memBuf);
        }

        //-----------------------------------------------------------------------------
        //! Gets the native pointer of the memory buffer.
        //!
        //! \param memBuf The memory buffer to fill.
        //! \return The native pointer.
        //-----------------------------------------------------------------------------
        template<
            typename TMemBuf>
        ALPAKA_FCT_HOST auto getNativePtr(
            TMemBuf & memBuf)
            -> GetMemElemT<TMemBuf> *
        {
            return traits::mem::GetNativePtr<TMemBuf>::getNativePtr(memBuf);
        }

        //-----------------------------------------------------------------------------
        //! \return The pitch in bytes. This is the distance between two consecutive rows.
        //-----------------------------------------------------------------------------
        template<
            typename TMemBuf>
        ALPAKA_FCT_HOST std::size_t getPitchBytes(
            TMemBuf const & memBuf)
        {
            return traits::mem::GetPitchBytes<TMemBuf>::getPitchBytes(memBuf);
        }

        //-----------------------------------------------------------------------------
        //! \return The pitch in elements. This is the distance between two consecutive rows.
        //-----------------------------------------------------------------------------
        template<
            typename TMemBuf>
        ALPAKA_FCT_HOST std::size_t getPitchElements(
            TMemBuf const & memBuf)
        {
            assert((getPitchBytes(memBuf) % sizeof(GetMemElemT<TMemBuf>)) == 0u);
            return getPitchBytes(memBuf) / sizeof(GetMemElemT<TMemBuf>);
        }

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
            -> decltype(traits::mem::MemAlloc<TElem, dim::GetDimT<TExtents>, GetMemSpaceT<TMemSpace>>::memAlloc(std::declval<TExtents>()))
        {
            return traits::mem::MemAlloc<TElem, dim::GetDimT<TExtents>, GetMemSpaceT<TMemSpace>>::memAlloc(
                extents);
        }

        //-----------------------------------------------------------------------------
        //! Copies memory possibly between different memory spaces.
        //!
        //! \param memBufDst The destination memory buffer.
        //! \param memBufSrc The source memory buffer.
        //! \param extents The extents of the buffer to copy.
        //-----------------------------------------------------------------------------
        template<
            typename TMemBufDst, 
            typename TMemBufSrc, 
            typename TExtents>
        ALPAKA_FCT_HOST void copy(
            TMemBufDst & memBufDst, 
            TMemBufSrc const & memBufSrc, 
            TExtents const & extents)
        {
            static_assert(
                std::is_same<dim::GetDimT<TMemBufDst>, dim::GetDimT<TMemBufSrc>>::value,
                "The source and the destination buffers are required to have the same dimensionality!");
            static_assert(
                std::is_same<dim::GetDimT<TMemBufDst>, dim::GetDimT<TExtents>>::value,
                "The destination buffer and the extents are required to have the same dimensionality!");
            static_assert(
                std::is_same<GetMemElemT<TMemBufDst>, GetMemElemT<TMemBufSrc>>::value,
                "The source and the destination buffers are required to have the same element type!");

            traits::mem::MemCopy<dim::GetDimT<TMemBufDst>, GetMemSpaceT<TMemBufDst>, GetMemSpaceT<TMemBufSrc>>::memCopy(
                memBufDst,
                memBufSrc,
                extents);
        }

        //-----------------------------------------------------------------------------
        //! Copies memory possibly between different memory spaces asynchronously.
        //!
        //! \param memBufDst The destination memory buffer.
        //! \param memBufSrc The source memory buffer.
        //! \param extents The extents of the buffer to copy.
        //-----------------------------------------------------------------------------
        template<
            typename TMemBufDst, 
            typename TMemBufSrc, 
            typename TExtents,
            typename TStream>
        ALPAKA_FCT_HOST void copy(
            TMemBufDst & memBufDst, 
            TMemBufSrc const & memBufSrc, 
            TExtents const & extents,
            TStream const & stream)
        {
            static_assert(
                std::is_same<dim::GetDimT<TMemBufDst>, dim::GetDimT<TMemBufSrc>>::value,
                "The source and the destination buffers are required to have the same dimensionality!");
            static_assert(
                std::is_same<dim::GetDimT<TMemBufDst>, dim::GetDimT<TExtents>>::value,
                "The destination buffer and the extents are required to have the same dimensionality!");
            static_assert(
                std::is_same<GetMemElemT<TMemBufDst>, GetMemElemT<TMemBufSrc>>::value,
                "The source and the destination buffers are required to have the same element type!");

            traits::mem::MemCopy<dim::GetDimT<TMemBufDst>, GetMemSpaceT<TMemBufDst>, GetMemSpaceT<TMemBufSrc>>::memCopy(
                memBufDst,
                memBufSrc,
                extents,
                stream);
        }

        //-----------------------------------------------------------------------------
        //! Sets the memory to the given value.
        //!
        //! \param memBuf The memory buffer to fill.
        //! \param byte Value to set for each element of the specified buffer.
        //! \param extents The extents of the buffer to fill.
        //-----------------------------------------------------------------------------
        template<
            typename TMemBuf, 
            typename TExtents>
        ALPAKA_FCT_HOST void set(
            TMemBuf & memBuf, 
            std::uint8_t const & byte, 
            TExtents const & extents)
        {
            static_assert(
                std::is_same<dim::GetDimT<TMemBuf>, dim::GetDimT<TExtents>>::value,
                "The buffer and the extents are required to have the same dimensionality!");

            traits::mem::MemSet<dim::GetDimT<TMemBuf>, GetMemSpaceT<TMemBuf>>::memSet(
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
            typename TMemBuf, 
            typename TExtents,
            typename TStream>
        ALPAKA_FCT_HOST void set(
            TMemBuf & memBuf, 
            std::uint8_t const & byte, 
            TExtents const & extents,
            TStream const & stream)
        {
            static_assert(
                std::is_same<dim::GetDimT<TMemBuf>, dim::GetDimT<TExtents>>::value,
                "The buffer and the extents are required to have the same dimensionality!");

            traits::mem::MemSet<dim::GetDimT<TMemBuf>, GetMemSpaceT<TMemBuf>, TStream>::memSetAsync(
                memBuf,
                byte,
                extents,
                stream);
        }
    }
}
