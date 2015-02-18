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
#include <alpaka/traits/Extents.hpp>        // traits::getXXX
#include <alpaka/traits/mem/MemSpace.hpp>   // MemSpaceT
#include <alpaka/traits/mem/MemBufBase.hpp> // MemAlloc
#include <alpaka/traits/mem/MemBufView.hpp> // MemBufViewType

namespace alpaka
{
    namespace traits
    {
        namespace mem
        {
            //#############################################################################
            //! The memory element type trait.
            //#############################################################################
            template<
                typename TMemBuf, 
                typename TSfinae = void>
            struct MemElemType;

            //#############################################################################
            //! The base buffer trait.
            //#############################################################################
            template<
                typename TMemBufView, 
                typename TSfinae = void>
            struct GetMemBufBase;

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
        }
    }

    namespace mem
    {
        //#############################################################################
        //! The memory element type trait alias template to remove the ::type.
        //#############################################################################
        template<
            typename TMemBuf>
        using MemElemT = typename traits::mem::MemElemType<TMemBuf>::type;

        //-----------------------------------------------------------------------------
        //! Gets the base memory buffer.
        //!
        //! \param memBuf The memory buffer.
        //! \return The base buffer.
        //-----------------------------------------------------------------------------
        template<
            typename TMemBuf>
        ALPAKA_FCT_HOST auto getMemBufBase(
            TMemBuf const & memBuf)
            -> decltype(traits::mem::GetMemBufBase<TMemBuf>::getMemBufBase(std::declval<TMemBuf const &>()))
        {
            return traits::mem::GetMemBufBase<TMemBuf>::getMemBufBase(memBuf);
        }

        //-----------------------------------------------------------------------------
        //! Gets the base memory buffer.
        //!
        //! \param memBuf The memory buffer.
        //! \return The base buffer.
        //-----------------------------------------------------------------------------
        template<
            typename TMemBuf>
        ALPAKA_FCT_HOST auto getMemBufBase(
            TMemBuf & memBuf)
            -> decltype(traits::mem::GetMemBufBase<TMemBuf>::getMemBufBase(std::declval<TMemBuf &>()))
        {
            return traits::mem::GetMemBufBase<TMemBuf>::getMemBufBase(memBuf);
        }

        //-----------------------------------------------------------------------------
        //! Gets the native pointer of the memory buffer.
        //!
        //! \param memBuf The memory buffer.
        //! \return The native pointer.
        //-----------------------------------------------------------------------------
        template<
            typename TMemBuf>
        ALPAKA_FCT_HOST auto getNativePtr(
            TMemBuf const & memBuf)
            -> MemElemT<TMemBuf> const *
        {
            return traits::mem::GetNativePtr<TMemBuf>::getNativePtr(memBuf);
        }

        //-----------------------------------------------------------------------------
        //! Gets the native pointer of the memory buffer.
        //!
        //! \param memBuf The memory buffer.
        //! \return The native pointer.
        //-----------------------------------------------------------------------------
        template<
            typename TMemBuf>
        ALPAKA_FCT_HOST auto getNativePtr(
            TMemBuf & memBuf)
            -> MemElemT<TMemBuf> *
        {
            return traits::mem::GetNativePtr<TMemBuf>::getNativePtr(memBuf);
        }

        //-----------------------------------------------------------------------------
        //! \return The pitch in bytes. This is the distance between two consecutive rows.
        //-----------------------------------------------------------------------------
        template<
            typename TMemBuf>
        ALPAKA_FCT_HOST UInt getPitchBytes(
            TMemBuf const & memBuf)
        {
            return traits::mem::GetPitchBytes<TMemBuf>::getPitchBytes(memBuf);
        }

        //-----------------------------------------------------------------------------
        //! \return The pitch in elements. This is the distance between two consecutive rows.
        //-----------------------------------------------------------------------------
        template<
            typename TMemBuf>
        ALPAKA_FCT_HOST UInt getPitchElements(
            TMemBuf const & memBuf)
        {
            assert((getPitchBytes(memBuf) % sizeof(MemElemT<TMemBuf>)) == 0u);
            return getPitchBytes(memBuf) / sizeof(MemElemT<TMemBuf>);
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
                std::is_same<dim::DimT<TMemBufDst>, dim::DimT<TMemBufSrc>>::value,
                "The source and the destination buffers are required to have the same dimensionality!");
            static_assert(
                std::is_same<dim::DimT<TMemBufDst>, dim::DimT<TExtents>>::value,
                "The destination buffer and the extents are required to have the same dimensionality!");
            static_assert(
                std::is_same<MemElemT<TMemBufDst>, MemElemT<TMemBufSrc>>::value,
                "The source and the destination buffers are required to have the same element type!");

            traits::mem::MemCopy<dim::DimT<TMemBufDst>, MemSpaceT<TMemBufDst>, MemSpaceT<TMemBufSrc>>::memCopy(
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
                std::is_same<dim::DimT<TMemBufDst>, dim::DimT<TMemBufSrc>>::value,
                "The source and the destination buffers are required to have the same dimensionality!");
            static_assert(
                std::is_same<dim::DimT<TMemBufDst>, dim::DimT<TExtents>>::value,
                "The destination buffer and the extents are required to have the same dimensionality!");
            static_assert(
                std::is_same<MemElemT<TMemBufDst>, MemElemT<TMemBufSrc>>::value,
                "The source and the destination buffers are required to have the same element type!");

            traits::mem::MemCopy<dim::DimT<TMemBufDst>, MemSpaceT<TMemBufDst>, MemSpaceT<TMemBufSrc>>::memCopy(
                memBufDst,
                memBufSrc,
                extents,
                stream);
        }
    }
}
