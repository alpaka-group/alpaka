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

#include <alpaka/dim/DimIntegralConst.hpp>  // dim::DimInt<N>
#include <alpaka/extent/Traits.hpp>         // extent::getXXX
#include <alpaka/mem/view/Traits.hpp>       // view::Copy, ...
#include <alpaka/stream/StreamCpuAsync.hpp> // stream::StreamCpuAsync
#include <alpaka/stream/StreamCpuSync.hpp>  // stream::StreamCpuSync

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
            namespace cpu
            {
                namespace detail
                {
                    //#############################################################################
                    //! The CPU device memory copy task.
                    //!
                    //! Copies from CPU memory into CPU memory.
                    //!
                    //! TODO: Specialize for different dimensionalities to optimize.
                    //#############################################################################
                    template<
                        typename TBufDst,
                        typename TBufSrc,
                        typename TExtents>
                    struct TaskCopy
                    {
                        using Size = size::Size<TExtents>;

                        static_assert(
                            dim::Dim<TBufDst>::value == dim::Dim<TBufSrc>::value,
                            "The source and the destination buffers are required to have the same dimensionality!");
                        static_assert(
                            dim::Dim<TBufDst>::value == dim::Dim<TExtents>::value,
                            "The buffers and the extents are required to have the same dimensionality!");
                        // TODO: Maybe check for Size of TBufDst and TBufSrc to have greater or equal range than TExtents.
                        static_assert(
                            std::is_same<mem::view::Elem<TBufDst>, typename std::remove_const<mem::view::Elem<TBufSrc>>::type>::value,
                            "The source and the destination buffers are required to have the same element type!");

                        //-----------------------------------------------------------------------------
                        //! Constructor.
                        //-----------------------------------------------------------------------------
                        TaskCopy(
                            TBufDst & bufDst,
                            TBufSrc const & bufSrc,
                            TExtents const & extents) :
                                m_extentWidth(extent::getWidth(extents)),
                                m_extentWidthBytes(static_cast<Size>(m_extentWidth * sizeof(mem::view::Elem<TBufDst>))),
                                m_dstWidth(static_cast<Size>(extent::getWidth(bufDst))),
                                m_srcWidth(static_cast<Size>(extent::getWidth(bufSrc))),
                                m_dstBufWidth(static_cast<Size>(extent::getWidth(mem::view::getBuf(bufDst)))),
                                m_srcBufWidth(static_cast<Size>(extent::getWidth(mem::view::getBuf(bufSrc)))),

                                m_extentHeight(extent::getHeight(extents)),
                                m_dstHeight(static_cast<Size>(extent::getHeight(bufDst))),
                                m_srcHeight(static_cast<Size>(extent::getHeight(bufSrc))),
                                m_dstBufHeight(static_cast<Size>(extent::getHeight(mem::view::getBuf(bufDst)))),
                                m_srcBufHeight(static_cast<Size>(extent::getHeight(mem::view::getBuf(bufSrc)))),

                                m_extentDepth(extent::getDepth(extents)),
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                                m_dstDepth(static_cast<Size>(extent::getDepth(bufDst))),
                                m_srcDepth(static_cast<Size>(extent::getDepth(bufSrc))),
#endif
                                m_dstPitchBytes(static_cast<Size>(mem::view::getPitchBytes<dim::Dim<TBufDst>::value - 1u>(bufDst))),
                                m_srcPitchBytes(static_cast<Size>(mem::view::getPitchBytes<dim::Dim<TBufSrc>::value - 1u>(bufSrc))),

                                m_dstMemNative(reinterpret_cast<std::uint8_t *>(mem::view::getPtrNative(bufDst))),
                                m_srcMemNative(reinterpret_cast<std::uint8_t const *>(mem::view::getPtrNative(bufSrc)))
                        {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            assert(m_extentWidth <= m_dstWidth);
                            assert(m_extentHeight <= m_dstHeight);
                            assert(m_extentDepth <= m_dstDepth);
                            assert(m_extentWidth <= m_srcWidth);
                            assert(m_extentHeight <= m_srcHeight);
                            assert(m_extentDepth <= m_srcDepth);
                            assert(m_extentWidthBytes <= m_dstPitchBytes);
                            assert(m_extentWidthBytes <= m_srcPitchBytes);
#endif
                        }

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST auto printDebug() const
                        -> void
                        {
                            std::cout << BOOST_CURRENT_FUNCTION
                                << " ew: " << m_extentWidth
                                << " eh: " << m_extentHeight
                                << " ed: " << m_extentDepth
                                << " ewb: " << m_extentWidthBytes
                                << " dw: " << m_dstWidth
                                << " dh: " << m_dstHeight
                                << " dd: " << m_dstDepth
                                << " dptr: " << m_dstMemNative
                                << " dpitchb: " << m_dstPitchBytes
                                << " dbasew: " << m_dstBufWidth
                                << " dbaseh: " << m_dstBufHeight
                                << " sw: " << m_srcWidth
                                << " sh: " << m_srcHeight
                                << " sd: " << m_srcDepth
                                << " sptr: " << m_srcMemNative
                                << " spitchb: " << m_srcPitchBytes
                                << " sbasew: " << m_srcBufWidth
                                << " sbaseh: " << m_srcBufHeight
                                << std::endl;
                        }
#endif
                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST auto operator()() const
                        -> void
                        {
                            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                            auto const dstSliceSizeBytes(m_dstPitchBytes * m_dstHeight);
                            auto const srcSliceSizeBytes(m_srcPitchBytes * m_srcHeight);

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            printDebug();
#endif
                            auto const equalWidths(
                                (m_extentWidth == m_dstWidth)
                                && (m_extentWidth == m_srcWidth)
                                && (m_extentWidth == m_dstBufWidth)
                                && (m_extentWidth == m_srcBufWidth));

                            // If:
                            // - the copy extents width is identical to the dst and src extents width
                            // - the copy extents width is identical to the dst and src memory buffer extents width
                            // - the src and dst pitch is identical
                            // -> we can copy whole slices at once overwriting the pitch bytes
                            auto const copySliceAtOnce(
                                equalWidths
                                && (m_dstPitchBytes == m_srcPitchBytes));

                            // If:
                            // - the copy extents width and height are identical to the dst and src extents width and height
                            // - the copy extents width and height are identical to the dst and src memory buffer extents width and height
                            // - the src and dst slice size is identical
                            // -> we can copy the whole memory at once overwriting the pitch bytes
                            auto const copyAllAtOnce(
                                (m_extentHeight == m_dstHeight)
                                && (m_extentHeight == m_srcHeight)
                                && (m_extentHeight == m_dstBufHeight)
                                && (m_extentHeight == m_srcBufHeight)
                                && (dstSliceSizeBytes == srcSliceSizeBytes)
                                && copySliceAtOnce);

                            if(copyAllAtOnce)
                            {
                                std::memcpy(
                                    reinterpret_cast<void *>(m_dstMemNative),
                                    reinterpret_cast<void const *>(m_srcMemNative),
                                    dstSliceSizeBytes*m_extentDepth);
                            }
                            else
                            {
                                for(auto z(decltype(m_extentDepth)(0)); z < m_extentDepth; ++z)
                                {
                                    if(copySliceAtOnce)
                                    {
                                        std::memcpy(
                                            reinterpret_cast<void *>(m_dstMemNative + z*dstSliceSizeBytes),
                                            reinterpret_cast<void const *>(m_srcMemNative + z*srcSliceSizeBytes),
                                            m_dstPitchBytes*m_extentHeight);
                                    }
                                    else
                                    {
                                        for(auto y((decltype(m_extentHeight)(0))); y < m_extentHeight; ++y)
                                        {
                                            std::memcpy(
                                                reinterpret_cast<void *>(m_dstMemNative + y*m_dstPitchBytes + z*dstSliceSizeBytes),
                                                reinterpret_cast<void const *>(m_srcMemNative + y*m_srcPitchBytes + z*srcSliceSizeBytes),
                                                m_extentWidthBytes);
                                        }
                                    }
                                }
                            }
                        }

                        Size m_extentWidth;
                        Size m_extentWidthBytes;
                        Size m_dstWidth;
                        Size m_srcWidth;
                        Size m_dstBufWidth;
                        Size m_srcBufWidth;

                        Size m_extentHeight;
                        Size m_dstHeight;
                        Size m_srcHeight;
                        Size m_dstBufHeight;
                        Size m_srcBufHeight;

                        Size m_extentDepth;
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        Size m_dstDepth;
                        Size m_srcDepth;
#endif
                        Size m_dstPitchBytes;
                        Size m_srcPitchBytes;

                        std::uint8_t * m_dstMemNative;
                        std::uint8_t const * m_srcMemNative;
                    };
                }
            }

            namespace traits
            {
                //#############################################################################
                //! The CPU device memory copy trait specialization.
                //!
                //! Copies from CPU memory into CPU memory.
                //#############################################################################
                template<
                    typename TDim>
                struct TaskCopy<
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
                    ALPAKA_FN_HOST static auto taskCopy(
                        TBufDst & bufDst,
                        TBufSrc const & bufSrc,
                        TExtents const & extents)
                    -> cpu::detail::TaskCopy<
                        TBufDst,
                        TBufSrc,
                        TExtents>
                    {
                        return
                            cpu::detail::TaskCopy<
                                TBufDst,
                                TBufSrc,
                                TExtents>(
                                    bufDst,
                                    bufSrc,
                                    extents);
                    }
                };
            }
        }
    }
}
