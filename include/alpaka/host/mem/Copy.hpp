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

#include <alpaka/host/mem/Space.hpp>    // SpaceHost
#include <alpaka/host/mem/Buf.hpp>      // BufHost
#include <alpaka/host/Stream.hpp>       // StreamHost

#include <alpaka/core/BasicDims.hpp>    // dim::Dim<N>

#include <alpaka/traits/Mem.hpp>        // traits::Copy, ...
#include <alpaka/traits/Extent.hpp>     // traits::getXXX

#include <cassert>                      // assert
#include <cstring>                      // std::memcpy

namespace alpaka
{
    //-----------------------------------------------------------------------------
    // Trait specializations for host::detail::BufHost.
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
            struct Copy<
                TDim, 
                alpaka::mem::SpaceHost,
                alpaka::mem::SpaceHost>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents, 
                    typename TBufSrc, 
                    typename TBufDst>
                ALPAKA_FCT_HOST static auto copy(
                    TBufDst & bufDst, 
                    TBufSrc const & bufSrc, 
                    TExtents const & extents)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    static_assert(
                        std::is_same<alpaka::dim::DimT<TBufDst>, alpaka::dim::DimT<TBufSrc>>::value,
                        "The source and the destination buffers are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<alpaka::dim::DimT<TBufDst>, alpaka::dim::DimT<TExtents>>::value,
                        "The buffers and the extents are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<alpaka::mem::ElemT<TBufDst>, alpaka::mem::ElemT<TBufSrc>>::value,
                        "The source and the destination buffers are required to have the same element type!");

                    using Elem = alpaka::mem::ElemT<TBufDst>;

                    auto const uiExtentWidth(alpaka::extent::getWidth(extents));
                    auto const uiExtentHeight(alpaka::extent::getHeight(extents));
                    auto const uiExtentDepth(alpaka::extent::getDepth(extents));
                    auto const uiDstWidth(alpaka::extent::getWidth(bufDst));
                    auto const uiDstHeight(alpaka::extent::getHeight(bufDst));
#if (!defined(NDEBUG)) || (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                    auto const uiDstDepth(alpaka::extent::getDepth(bufDst));
#endif
                    auto const uiSrcWidth(alpaka::extent::getWidth(bufSrc));
                    auto const uiSrcHeight(alpaka::extent::getHeight(bufSrc));
#if (!defined(NDEBUG)) || (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                    auto const uiSrcDepth(alpaka::extent::getDepth(bufSrc));
#endif
                    assert(uiExtentWidth <= uiDstWidth);
                    assert(uiExtentHeight <= uiDstHeight);
                    assert(uiExtentDepth <= uiDstDepth);
                    assert(uiExtentWidth <= uiSrcWidth);
                    assert(uiExtentHeight <= uiSrcHeight);
                    assert(uiExtentDepth <= uiSrcDepth);

                    auto const uiExtentWidthBytes(uiExtentWidth * sizeof(Elem));
                    auto const uiDstPitchBytes(alpaka::mem::getPitchBytes(bufDst));
                    auto const uiSrcPitchBytes(alpaka::mem::getPitchBytes(bufSrc));
                    assert(uiExtentWidthBytes <= uiDstPitchBytes);
                    assert(uiExtentWidthBytes <= uiSrcPitchBytes);

                    auto const pDstNative(reinterpret_cast<std::uint8_t *>(alpaka::mem::getNativePtr(bufDst)));
                    auto const pSrcNative(reinterpret_cast<std::uint8_t const *>(alpaka::mem::getNativePtr(bufSrc)));
                    auto const uiDstSliceSizeBytes(uiDstPitchBytes * uiDstHeight);
                    auto const uiSrcSliceSizeBytes(uiSrcPitchBytes * uiSrcHeight);
                    
                    auto const & dstBuf(alpaka::mem::getBuf(bufDst));
                    auto const & srcBuf(alpaka::mem::getBuf(bufSrc));
                    auto const uiDstBufWidth(alpaka::extent::getWidth(dstBuf));
                    auto const uiSrcBufWidth(alpaka::extent::getWidth(srcBuf));
                    auto const uiDstBufHeight(alpaka::extent::getHeight(dstBuf));
                    auto const uiSrcBufHeight(alpaka::extent::getHeight(srcBuf));
                    
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    std::cout << BOOST_CURRENT_FUNCTION
                        << " ew: " << uiExtentWidth
                        << " eh: " << uiExtentHeight
                        << " ed: " << uiExtentDepth
                        << " ewb: " << uiExtentWidthBytes
                        << " dw: " << uiDstWidth
                        << " dh: " << uiDstHeight
                        << " dd: " << uiDstDepth
                        << " dptr: " << reinterpret_cast<void *>(pDstNative)
                        << " dpitchb: " << uiDstPitchBytes
                        << " dbasew: " << uiDstBufWidth
                        << " dbaseh: " << uiDstBufHeight
                        << " sw: " << uiSrcWidth
                        << " sh: " << uiSrcHeight
                        << " sd: " << uiSrcDepth
                        << " sptr: " << reinterpret_cast<void const *>(pSrcNative)
                        << " spitchb: " << uiSrcPitchBytes
                        << " sbasew: " << uiSrcBufWidth
                        << " sbaseh: " << uiSrcBufHeight
                        << std::endl;
#endif
                    // If:
                    // - the copy extents width and height are identical to the dst and src extents width and height
                    // - the copy extents width and height are identical to the dst and src base memory buffer extents width and height
                    // - the src and dst slice size is identical 
                    // -> we can copy the whole memory at once overwriting the pitch bytes
                    if((uiExtentWidth == uiDstWidth)
                        && (uiExtentWidth == uiSrcWidth)
                        && (uiExtentHeight == uiDstHeight)
                        && (uiExtentHeight == uiSrcHeight)
                        && (uiExtentWidth == uiDstBufWidth)
                        && (uiExtentWidth == uiSrcBufWidth)
                        && (uiExtentHeight == uiDstBufHeight)
                        && (uiExtentHeight == uiSrcBufHeight)
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
                                && (uiExtentWidth == uiDstBufWidth)
                                && (uiExtentWidth == uiSrcBufWidth)
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
                    typename TBufSrc, 
                    typename TBufDst>
                ALPAKA_FCT_HOST static auto copy(
                    TBufDst & bufDst, 
                    TBufSrc const & bufSrc, 
                    TExtents const & extents,
                    host::detail::StreamHost const &)
                -> void
                {
                    // \TODO: Implement asynchronous host copy.
                    copy(
                        bufDst,
                        bufSrc,
                        extents);
                }
            };
        }
    }
}
