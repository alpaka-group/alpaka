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

#include <alpaka/host/MemSpace.hpp>         // MemSpaceHost
#include <alpaka/host/mem/MemBufBase.hpp>   // MemBufBaseHost
#include <alpaka/host/Stream.hpp>           // StreamHost

#include <alpaka/core/BasicDims.hpp>        // dim::Dim<N>

#include <alpaka/traits/Mem.hpp>            // traits::MemCopy, ...
#include <alpaka/traits/Extents.hpp>        // traits::getXXX

#include <cassert>                          // assert
#include <cstring>                          // std::memcpy

namespace alpaka
{
    //-----------------------------------------------------------------------------
    // Trait specializations for host::detail::MemBufBaseHost.
    //-----------------------------------------------------------------------------
    namespace traits
    {
        namespace mem
        {
            //#############################################################################
            //! The host accelerators memory copy trait specialization.
            //!
            //! Copies from host memory into host memory.
            //#############################################################################
            template<
                typename TDim>
            struct MemCopy<
                TDim, 
                alpaka::mem::MemSpaceHost,
                alpaka::mem::MemSpaceHost>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents, 
                    typename TMemBufSrc, 
                    typename TMemBufDst>
                ALPAKA_FCT_HOST static void memCopy(
                    TMemBufDst & memBufDst, 
                    TMemBufSrc const & memBufSrc, 
                    TExtents const & extents)
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    static_assert(
                        std::is_same<alpaka::dim::DimT<TMemBufDst>, alpaka::dim::DimT<TMemBufSrc>>::value,
                        "The source and the destination buffers are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<alpaka::dim::DimT<TMemBufDst>, alpaka::dim::DimT<TExtents>>::value,
                        "The buffers and the extents are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<alpaka::mem::MemElemT<TMemBufDst>, alpaka::mem::MemElemT<TMemBufSrc>>::value,
                        "The source and the destination buffers are required to have the same element type!");

                    using Elem = alpaka::mem::MemElemT<TMemBufDst>;

                    auto const uiExtentWidth(alpaka::extent::getWidth(extents));
                    auto const uiExtentHeight(alpaka::extent::getHeight(extents));
                    auto const uiExtentDepth(alpaka::extent::getDepth(extents));
                    auto const uiDstWidth(alpaka::extent::getWidth(memBufDst));
                    auto const uiDstHeight(alpaka::extent::getHeight(memBufDst));
#ifndef NDEBUG
                    auto const uiDstDepth(alpaka::extent::getDepth(memBufDst));
#endif
                    auto const uiSrcWidth(alpaka::extent::getWidth(memBufSrc));
                    auto const uiSrcHeight(alpaka::extent::getHeight(memBufSrc));
#ifndef NDEBUG
                    auto const uiSrcDepth(alpaka::extent::getDepth(memBufSrc));
#endif
                    assert(uiExtentWidth <= uiDstWidth);
                    assert(uiExtentHeight <= uiDstHeight);
                    assert(uiExtentDepth <= uiDstDepth);
                    assert(uiExtentWidth <= uiSrcWidth);
                    assert(uiExtentHeight <= uiSrcHeight);
                    assert(uiExtentDepth <= uiSrcDepth);

                    auto const uiExtentWidthBytes(uiExtentWidth * sizeof(Elem));
                    auto const uiDstPitchBytes(alpaka::mem::getPitchBytes(memBufDst));
                    auto const uiSrcPitchBytes(alpaka::mem::getPitchBytes(memBufSrc));
                    assert(uiExtentWidthBytes <= uiDstPitchBytes);
                    assert(uiExtentWidthBytes <= uiSrcPitchBytes);

                    auto const pDstNative(reinterpret_cast<std::uint8_t *>(alpaka::mem::getNativePtr(memBufDst)));
                    auto const pSrcNative(reinterpret_cast<std::uint8_t const *>(alpaka::mem::getNativePtr(memBufSrc)));
                    auto const uiDstSliceSizeBytes(uiDstPitchBytes * uiDstHeight);
                    auto const uiSrcSliceSizeBytes(uiSrcPitchBytes * uiSrcHeight);
                    
                    auto const & dstMemBufBase(alpaka::mem::getMemBufBase(memBufDst));
                    auto const & srcMemBufBase(alpaka::mem::getMemBufBase(memBufSrc));
                    auto const uiDstMemBufBaseWidth(alpaka::extent::getWidth(dstMemBufBase));
                    auto const uiSrcMemBufBaseWidth(alpaka::extent::getWidth(srcMemBufBase));
                    auto const uiDstMemBufBaseHeight(alpaka::extent::getHeight(dstMemBufBase));
                    auto const uiSrcMemBufBaseHeight(alpaka::extent::getHeight(srcMemBufBase));

                    // If:
                    // - the copy extents width and height are identical to the dst and src extents width and height
                    // - the copy extents width and height are identical to the dst and src base memory buffer extents width and height
                    // - the src and dst slice size is identical 
                    // -> we can copy the whole memory at once overwriting the pitch bytes
                    if((uiExtentWidth == uiDstWidth)
                        && (uiExtentWidth == uiSrcWidth)
                        && (uiExtentHeight == uiDstHeight)
                        && (uiExtentHeight == uiSrcHeight)
                        && (uiExtentWidth == uiDstMemBufBaseWidth)
                        && (uiExtentWidth == uiSrcMemBufBaseWidth)
                        && (uiExtentHeight == uiDstMemBufBaseHeight)
                        && (uiExtentHeight == uiSrcMemBufBaseHeight)
                        && (uiDstSliceSizeBytes == uiSrcSliceSizeBytes))
                    {
                        std::memcpy(
                            reinterpret_cast<void *>(pDstNative),
                            reinterpret_cast<void const *>(pSrcNative),
                            uiDstSliceSizeBytes*uiExtentDepth);
                    }
                    else
                    {
                        for(UInt z(0); z < uiExtentDepth; ++z)
                        {
                            // If:
                            // - the copy extents width is identical to the dst and src extents width
                            // - the copy extents width is identical to the dst and src base memory buffer extents width
                            // - the src and dst pitch is identical 
                            // -> we can copy whole slices at once overwriting the pitch bytes
                            if((uiExtentWidth == uiDstWidth)
                                && (uiExtentWidth == uiSrcWidth)
                                && (uiExtentWidth == uiDstMemBufBaseWidth)
                                && (uiExtentWidth == uiSrcMemBufBaseWidth)
                                && (uiDstPitchBytes == uiSrcPitchBytes))
                            {
                                std::memcpy(
                                    reinterpret_cast<void *>(pDstNative + z*uiDstSliceSizeBytes),
                                    reinterpret_cast<void const *>(pSrcNative + z*uiSrcSliceSizeBytes),
                                    uiDstPitchBytes*uiExtentHeight);
                            }
                            else
                            {
                                for(UInt y(0); y < uiExtentHeight; ++y)
                                {
                                    std::memcpy(
                                        reinterpret_cast<void *>(pDstNative + y*uiDstPitchBytes + z*uiDstSliceSizeBytes),
                                        reinterpret_cast<void const *>(pSrcNative + y*uiSrcPitchBytes + z*uiSrcSliceSizeBytes),
                                        uiExtentWidthBytes);
                                }
                            }
                        }
                    }
                }
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents, 
                    typename TMemBufSrc, 
                    typename TMemBufDst>
                ALPAKA_FCT_HOST static void memCopy(
                    TMemBufDst & memBufDst, 
                    TMemBufSrc const & memBufSrc, 
                    TExtents const & extents,
                    host::detail::StreamHost const &)
                {
                    // \TODO: Implement asynchronous host memCopy.
                    memCopy(
                        memBufDst,
                        memBufSrc,
                        extents);
                }
            };
        }
    }
}
