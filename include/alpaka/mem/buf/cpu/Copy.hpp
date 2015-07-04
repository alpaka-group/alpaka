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

#include <alpaka/dim/DimIntegralConst.hpp>  // dim::Dim<N>
#include <alpaka/extent/Traits.hpp>         // extent::getXXX
#include <alpaka/mem/view/Traits.hpp>       // view::Copy, ...
#include <alpaka/stream/StreamCpuAsync.hpp> // StreamCpuAsync

#include <boost/core/ignore_unused.hpp>     // boost::ignore_unused

#include <cassert>                          // assert
#include <cstring>                          // std::memcpy

namespace alpaka
{
    namespace dev
    {
        class DevCpu;
    }
}

namespace alpaka
{
    namespace mem
    {
        namespace view
        {
            namespace traits
            {
                //#############################################################################
                //! The CPU device memory copy trait specialization.
                //!
                //! Copies from CPU memory into CPU memory.
                //#############################################################################
                template<
                    typename TDim>
                struct Copy<
                    TDim,
                    dev::DevCpu,
                    dev::DevCpu>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtents,
                        typename TBufSrc,
                        typename TBufDst>
                    ALPAKA_FN_HOST static auto copy(
                        TBufDst & bufDst,
                        TBufSrc const & bufSrc,
                        TExtents const & extents)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        static_assert(
                            dim::DimT<TBufDst>::value == dim::DimT<TBufSrc>::value,
                            "The source and the destination buffers are required to have the same dimensionality!");
                        static_assert(
                            dim::DimT<TBufDst>::value == dim::DimT<TExtents>::value,
                            "The buffers and the extents are required to have the same dimensionality!");
                        static_assert(
                            std::is_same<mem::view::ElemT<TBufDst>, typename std::remove_const<mem::view::ElemT<TBufSrc>>::type>::value,
                            "The source and the destination buffers are required to have the same element type!");

                        using Elem = mem::view::ElemT<TBufDst>;

                        auto const uiExtentWidth(extent::getWidth(extents));
                        auto const uiExtentHeight(extent::getHeight(extents));
                        auto const uiExtentDepth(extent::getDepth(extents));
                        auto const uiDstWidth(extent::getWidth(bufDst));
                        auto const uiDstHeight(extent::getHeight(bufDst));
#if (!defined(NDEBUG)) || (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                        auto const uiDstDepth(extent::getDepth(bufDst));
#endif
                        auto const uiSrcWidth(extent::getWidth(bufSrc));
                        auto const uiSrcHeight(extent::getHeight(bufSrc));
#if (!defined(NDEBUG)) || (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                        auto const uiSrcDepth(extent::getDepth(bufSrc));
#endif
                        assert(uiExtentWidth <= uiDstWidth);
                        assert(uiExtentHeight <= uiDstHeight);
                        assert(uiExtentDepth <= uiDstDepth);
                        assert(uiExtentWidth <= uiSrcWidth);
                        assert(uiExtentHeight <= uiSrcHeight);
                        assert(uiExtentDepth <= uiSrcDepth);

                        auto const uiExtentWidthBytes(uiExtentWidth * sizeof(Elem));
                        auto const uiDstPitchBytes(mem::view::getPitchBytes<dim::DimT<TBufDst>::value - 1u>(bufDst));
                        auto const uiSrcPitchBytes(mem::view::getPitchBytes<dim::DimT<TBufSrc>::value - 1u>(bufSrc));
                        assert(uiExtentWidthBytes <= uiDstPitchBytes);
                        assert(uiExtentWidthBytes <= uiSrcPitchBytes);

                        auto const pDstNative(reinterpret_cast<std::uint8_t *>(mem::view::getPtrNative(bufDst)));
                        auto const pSrcNative(reinterpret_cast<std::uint8_t const *>(mem::view::getPtrNative(bufSrc)));
                        auto const uiDstSliceSizeBytes(uiDstPitchBytes * uiDstHeight);
                        auto const uiSrcSliceSizeBytes(uiSrcPitchBytes * uiSrcHeight);

                        auto const & dstBuf(mem::view::getBuf(bufDst));
                        auto const & srcBuf(mem::view::getBuf(bufSrc));
                        auto const uiDstBufWidth(extent::getWidth(dstBuf));
                        auto const uiSrcBufWidth(extent::getWidth(srcBuf));
                        auto const uiDstBufHeight(extent::getHeight(dstBuf));
                        auto const uiSrcBufHeight(extent::getHeight(srcBuf));

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
                        // - the copy extents width and height are identical to the dst and src memory buffer extents width and height
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
                            for(auto z(decltype(uiExtentDepth)(0)); z < uiExtentDepth; ++z)
                            {
                                // If:
                                // - the copy extents width is identical to the dst and src extents width
                                // - the copy extents width is identical to the dst and src memory buffer extents width
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
                                    for(auto y((decltype(uiExtentHeight)(0))); y < uiExtentHeight; ++y)
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
                    ALPAKA_FN_HOST static auto copy(
                        TBufDst & bufDst,
                        TBufSrc const & bufSrc,
                        TExtents const & extents,
                        stream::StreamCpuAsync const & stream)
                    -> void
                    {
                        boost::ignore_unused(stream);

                        stream.m_spAsyncStreamCpu->m_workerThread.enqueueTask(
                            [&bufDst, &bufSrc, extents]()
                            {
                                copy(
                                    bufDst,
                                    bufSrc,
                                    extents);
                            });
                    }
                };
            }
        }
    }
}
