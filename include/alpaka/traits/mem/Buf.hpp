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

#include <alpaka/traits/Acc.hpp>        // AccT
#include <alpaka/traits/Dim.hpp>        // DimT
#include <alpaka/traits/Extent.hpp>     // traits::getXXX
#include <alpaka/traits/mem/Space.hpp>  // SpaceT
#include <alpaka/traits/mem/Buf.hpp>    // Alloc
#include <alpaka/traits/mem/View.hpp>   // ViewType

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
                typename TBuf,
                typename TSfinae = void>
            struct ElemType;

            //#############################################################################
            //! The base buffer trait.
            //#############################################################################
            template<
                typename TBuf,
                typename TSfinae = void>
            struct GetBuf;

            //#############################################################################
            //! The native pointer get trait.
            //#############################################################################
            template<
                typename TBuf,
                typename TSfinae = void>
            struct GetNativePtr;

            //#############################################################################
            //! The pitch in bytes. This is the distance between two consecutive rows.
            //#############################################################################
            template<
                typename TBuf,
                typename TSfinae = void>
            struct GetPitchBytes;

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

            //#############################################################################
            //! The memory set trait.
            //!
            //! Fills the buffer with data.
            //#############################################################################
            template<
                typename TDim,
                typename TSpace,
                typename TSfinae = void>
            struct Set;

            //#############################################################################
            //! The memory copy trait.
            //!
            //! Copies memory from one buffer into another buffer possibly in a different memory space.
            //#############################################################################
            template<
                typename TDim,
                typename TSpaceDst,
                typename TSpaceSrc,
                typename TSfinae = void>
            struct Copy;
        }
    }

    namespace mem
    {
        //#############################################################################
        //! The memory element type trait alias template to remove the ::type.
        //#############################################################################
        template<
            typename TBuf>
        using ElemT = typename traits::mem::ElemType<TBuf>::type;

        //-----------------------------------------------------------------------------
        //! Gets the base memory buffer.
        //!
        //! \param buf The memory buffer.
        //! \return The base buffer.
        //-----------------------------------------------------------------------------
        template<
            typename TBuf>
        ALPAKA_FCT_HOST auto getBuf(
            TBuf const & buf)
        -> decltype(traits::mem::GetBuf<TBuf>::getBuf(std::declval<TBuf const &>()))
        {
            return traits::mem::GetBuf<
                TBuf>
            ::getBuf(
                buf);
        }

        //-----------------------------------------------------------------------------
        //! Gets the base memory buffer.
        //!
        //! \param buf The memory buffer.
        //! \return The base buffer.
        //-----------------------------------------------------------------------------
        template<
            typename TBuf>
        ALPAKA_FCT_HOST auto getBuf(
            TBuf & buf)
        -> decltype(traits::mem::GetBuf<TBuf>::getBuf(std::declval<TBuf &>()))
        {
            return traits::mem::GetBuf<
                TBuf>
            ::getBuf(
                buf);
        }

        //-----------------------------------------------------------------------------
        //! Gets the native pointer of the memory buffer.
        //!
        //! \param buf The memory buffer.
        //! \return The native pointer.
        //-----------------------------------------------------------------------------
        template<
            typename TBuf>
        ALPAKA_FCT_HOST auto getNativePtr(
            TBuf const & buf)
        -> ElemT<TBuf> const *
        {
            return traits::mem::GetNativePtr<
                TBuf>
            ::getNativePtr(
                buf);
        }

        //-----------------------------------------------------------------------------
        //! Gets the native pointer of the memory buffer.
        //!
        //! \param buf The memory buffer.
        //! \return The native pointer.
        //-----------------------------------------------------------------------------
        template<
            typename TBuf>
        ALPAKA_FCT_HOST auto getNativePtr(
            TBuf & buf)
        -> ElemT<TBuf> *
        {
            return traits::mem::GetNativePtr<
                TBuf>
            ::getNativePtr(
                buf);
        }

        //-----------------------------------------------------------------------------
        //! \return The pitch in bytes. This is the distance between two consecutive rows.
        //-----------------------------------------------------------------------------
        template<
            typename TBuf>
        ALPAKA_FCT_HOST auto getPitchBytes(
            TBuf const & buf)
        -> UInt
        {
            return traits::mem::GetPitchBytes<
                TBuf>
            ::getPitchBytes(
                buf);
        }

        //-----------------------------------------------------------------------------
        //! \return The pitch in elements. This is the distance between two consecutive rows.
        //-----------------------------------------------------------------------------
        template<
            typename TBuf>
        ALPAKA_FCT_HOST auto getPitchElements(
            TBuf const & buf)
        -> UInt
        {
            assert((getPitchBytes(buf) % sizeof(ElemT<TBuf>)) == 0u);
            return getPitchBytes(buf) / sizeof(ElemT<TBuf>);
        }

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

        //-----------------------------------------------------------------------------
        //! Sets the memory to the given value.
        //!
        //! \param buf The memory buffer to fill.
        //! \param byte Value to set for each element of the specified buffer.
        //! \param extents The extents of the buffer to fill.
        //-----------------------------------------------------------------------------
        template<
            typename TBuf,
            typename TExtents>
        ALPAKA_FCT_HOST auto set(
            TBuf & buf,
            std::uint8_t const & byte,
            TExtents const & extents)
        -> void
        {
            static_assert(
                dim::DimT<TBuf>::value == dim::DimT<TExtents>::value,
                "The buffer and the extents are required to have the same dimensionality!");

            traits::mem::Set<
                dim::DimT<TBuf>,
                SpaceT<TBuf>>
            ::set(
                buf,
                byte,
                extents);
        }

        //-----------------------------------------------------------------------------
        //! Sets the memory to the given value asynchronously.
        //!
        //! \param buf The memory buffer to fill.
        //! \param byte Value to set for each element of the specified buffer.
        //! \param extents The extents of the buffer to fill.
        //! \param stream The stream to enqueue the buffer fill task into.
        //-----------------------------------------------------------------------------
        template<
            typename TBuf,
            typename TExtents,
            typename TStream>
        ALPAKA_FCT_HOST auto set(
            TBuf & buf,
            std::uint8_t const & byte,
            TExtents const & extents,
            TStream const & stream)
        -> void
        {
            static_assert(
                dim::DimT<TBuf>::value == dim::DimT<TExtents>::value,
                "The buffer and the extents are required to have the same dimensionality!");

            traits::mem::Set<
                dim::DimT<TBuf>,
                SpaceT<TBuf>,
                TStream>
            ::memSetAsync(
                buf,
                byte,
                extents,
                stream);
        }

        //-----------------------------------------------------------------------------
        //! Copies memory possibly between different memory spaces.
        //!
        //! \param bufDst The destination memory buffer.
        //! \param bufSrc The source memory buffer.
        //! \param extents The extents of the buffer to copy.
        //-----------------------------------------------------------------------------
        template<
            typename TBufDst,
            typename TBufSrc,
            typename TExtents>
        ALPAKA_FCT_HOST auto copy(
            TBufDst & bufDst,
            TBufSrc const & bufSrc,
            TExtents const & extents)
        -> void
        {
            static_assert(
                dim::DimT<TBufDst>::value == dim::DimT<TBufSrc>::value,
                "The source and the destination buffers are required to have the same dimensionality!");
            static_assert(
                dim::DimT<TBufDst>::value == dim::DimT<TExtents>::value,
                "The destination buffer and the extents are required to have the same dimensionality!");
            static_assert(
                std::is_same<ElemT<TBufDst>, ElemT<TBufSrc>>::value,
                "The source and the destination buffers are required to have the same element type!");

            traits::mem::Copy<
                dim::DimT<TBufDst>,
                SpaceT<TBufDst>,
                SpaceT<TBufSrc>>
            ::copy(
                bufDst,
                bufSrc,
                extents);
        }

        //-----------------------------------------------------------------------------
        //! Copies memory possibly between different memory spaces asynchronously.
        //!
        //! \param bufDst The destination memory buffer.
        //! \param bufSrc The source memory buffer.
        //! \param extents The extents of the buffer to copy.
        //! \param stream The stream to enqueue the buffer fill task into.
        //-----------------------------------------------------------------------------
        template<
            typename TBufDst,
            typename TBufSrc,
            typename TExtents,
            typename TStream>
        ALPAKA_FCT_HOST auto copy(
            TBufDst & bufDst,
            TBufSrc const & bufSrc,
            TExtents const & extents,
            TStream const & stream)
        -> void
        {
            static_assert(
                dim::DimT<TBufDst>::value == dim::DimT<TBufSrc>::value,
                "The source and the destination buffers are required to have the same dimensionality!");
            static_assert(
                dim::DimT<TBufDst>::value == dim::DimT<TExtents>::value,
                "The destination buffer and the extents are required to have the same dimensionality!");
            static_assert(
                std::is_same<ElemT<TBufDst>, ElemT<TBufSrc>>::value,
                "The source and the destination buffers are required to have the same element type!");

            traits::mem::Copy<
                dim::DimT<TBufDst>,
                SpaceT<TBufDst>,
                SpaceT<TBufSrc>>
            ::copy(
                bufDst,
                bufSrc,
                extents,
                stream);
        }
    }
}
