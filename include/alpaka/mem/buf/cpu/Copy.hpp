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
                                m_uiExtentWidth(extent::getWidth(extents)),
                                m_uiExtentWidthBytes(m_uiExtentWidth * sizeof(mem::view::Elem<TBufDst>)),
                                m_uiDstWidth(extent::getWidth(bufDst)),
                                m_uiSrcWidth(extent::getWidth(bufSrc)),
                                m_uiDstBufWidth(extent::getWidth(mem::view::getBuf(bufDst))),
                                m_uiSrcBufWidth(extent::getWidth(mem::view::getBuf(bufSrc))),

                                m_uiExtentHeight(extent::getHeight(extents)),
                                m_uiDstHeight(extent::getHeight(bufDst)),
                                m_uiSrcHeight(extent::getHeight(bufSrc)),
                                m_uiDstBufHeight(extent::getHeight(mem::view::getBuf(bufDst))),
                                m_uiSrcBufHeight(extent::getHeight(mem::view::getBuf(bufSrc))),

                                m_uiExtentDepth(extent::getDepth(extents)),
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                                m_uiDstDepth(extent::getDepth(bufDst)),
                                m_uiSrcDepth(extent::getDepth(bufSrc)),
#endif
                                m_uiDstPitchBytes(mem::view::getPitchBytes<dim::Dim<TBufDst>::value - 1u>(bufDst)),
                                m_uiSrcPitchBytes(mem::view::getPitchBytes<dim::Dim<TBufSrc>::value - 1u>(bufSrc)),

                                m_pDstNative(reinterpret_cast<std::uint8_t *>(mem::view::getPtrNative(bufDst))),
                                m_pSrcNative(reinterpret_cast<std::uint8_t const *>(mem::view::getPtrNative(bufSrc)))
                        {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            assert(m_uiExtentWidth <= m_uiDstWidth);
                            assert(m_uiExtentHeight <= m_uiDstHeight);
                            assert(m_uiExtentDepth <= m_uiDstDepth);
                            assert(m_uiExtentWidth <= m_uiSrcWidth);
                            assert(m_uiExtentHeight <= m_uiSrcHeight);
                            assert(m_uiExtentDepth <= m_uiSrcDepth);
                            assert(m_uiExtentWidthBytes <= m_uiDstPitchBytes);
                            assert(m_uiExtentWidthBytes <= m_uiSrcPitchBytes);
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
                                << " ew: " << m_uiExtentWidth
                                << " eh: " << m_uiExtentHeight
                                << " ed: " << m_uiExtentDepth
                                << " ewb: " << m_uiExtentWidthBytes
                                << " dw: " << m_uiDstWidth
                                << " dh: " << m_uiDstHeight
                                << " dd: " << m_uiDstDepth
                                << " dptr: " << m_pDstNative
                                << " dpitchb: " << m_uiDstPitchBytes
                                << " dbasew: " << m_uiDstBufWidth
                                << " dbaseh: " << m_uiDstBufHeight
                                << " sw: " << m_uiSrcWidth
                                << " sh: " << m_uiSrcHeight
                                << " sd: " << m_uiSrcDepth
                                << " sptr: " << m_pSrcNative
                                << " spitchb: " << m_uiSrcPitchBytes
                                << " sbasew: " << m_uiSrcBufWidth
                                << " sbaseh: " << m_uiSrcBufHeight
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

                            auto const uiDstSliceSizeBytes(m_uiDstPitchBytes * m_uiDstHeight);
                            auto const uiSrcSliceSizeBytes(m_uiSrcPitchBytes * m_uiSrcHeight);

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            printDebug();
#endif
                            auto const bEqualWidths(
                                (m_uiExtentWidth == m_uiDstWidth)
                                && (m_uiExtentWidth == m_uiSrcWidth)
                                && (m_uiExtentWidth == m_uiDstBufWidth)
                                && (m_uiExtentWidth == m_uiSrcBufWidth));

                            // If:
                            // - the copy extents width is identical to the dst and src extents width
                            // - the copy extents width is identical to the dst and src memory buffer extents width
                            // - the src and dst pitch is identical
                            // -> we can copy whole slices at once overwriting the pitch bytes
                            auto const bCopySlice(
                                bEqualWidths
                                && (m_uiDstPitchBytes == m_uiSrcPitchBytes));

                            // If:
                            // - the copy extents width and height are identical to the dst and src extents width and height
                            // - the copy extents width and height are identical to the dst and src memory buffer extents width and height
                            // - the src and dst slice size is identical
                            // -> we can copy the whole memory at once overwriting the pitch bytes
                            auto const bSingleCopy(
                                (m_uiExtentHeight == m_uiDstHeight)
                                && (m_uiExtentHeight == m_uiSrcHeight)
                                && (m_uiExtentHeight == m_uiDstBufHeight)
                                && (m_uiExtentHeight == m_uiSrcBufHeight)
                                && (uiDstSliceSizeBytes == uiSrcSliceSizeBytes)
                                && bCopySlice);

                            if(bSingleCopy)
                            {
                                std::memcpy(
                                    reinterpret_cast<void *>(m_pDstNative),
                                    reinterpret_cast<void const *>(m_pSrcNative),
                                    uiDstSliceSizeBytes*m_uiExtentDepth);
                            }
                            else
                            {
                                for(auto z(decltype(m_uiExtentDepth)(0)); z < m_uiExtentDepth; ++z)
                                {
                                    if(bCopySlice)
                                    {
                                        std::memcpy(
                                            reinterpret_cast<void *>(m_pDstNative + z*uiDstSliceSizeBytes),
                                            reinterpret_cast<void const *>(m_pSrcNative + z*uiSrcSliceSizeBytes),
                                            m_uiDstPitchBytes*m_uiExtentHeight);
                                    }
                                    else
                                    {
                                        for(auto y((decltype(m_uiExtentHeight)(0))); y < m_uiExtentHeight; ++y)
                                        {
                                            std::memcpy(
                                                reinterpret_cast<void *>(m_pDstNative + y*m_uiDstPitchBytes + z*uiDstSliceSizeBytes),
                                                reinterpret_cast<void const *>(m_pSrcNative + y*m_uiSrcPitchBytes + z*uiSrcSliceSizeBytes),
                                                m_uiExtentWidthBytes);
                                        }
                                    }
                                }
                            }
                        }

                        Size m_uiExtentWidth;
                        Size m_uiExtentWidthBytes;
                        Size m_uiDstWidth;
                        Size m_uiSrcWidth;
                        Size m_uiDstBufWidth;
                        Size m_uiSrcBufWidth;

                        Size m_uiExtentHeight;
                        Size m_uiDstHeight;
                        Size m_uiSrcHeight;
                        Size m_uiDstBufHeight;
                        Size m_uiSrcBufHeight;

                        Size m_uiExtentDepth;
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        Size m_uiDstDepth;
                        Size m_uiSrcDepth;
#endif
                        Size m_uiDstPitchBytes;
                        Size m_uiSrcPitchBytes;

                        std::uint8_t * m_pDstNative;
                        std::uint8_t const * m_pSrcNative;
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
