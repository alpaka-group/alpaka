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
#include <alpaka/mem/view/Traits.hpp>       // mem::view::Copy, ...

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
                        typename TViewDst,
                        typename TViewSrc,
                        typename TExtent>
                    struct TaskCopy
                    {
                        using Size = size::Size<TExtent>;

                        static_assert(
                            dim::Dim<TViewDst>::value == dim::Dim<TViewSrc>::value,
                            "The source and the destination view are required to have the same dimensionality!");
                        static_assert(
                            dim::Dim<TViewDst>::value == dim::Dim<TExtent>::value,
                            "The views and the extent are required to have the same dimensionality!");
                        // TODO: Maybe check for Size of TViewDst and TViewSrc to have greater or equal range than TExtent.
                        static_assert(
                            std::is_same<elem::Elem<TViewDst>, typename std::remove_const<elem::Elem<TViewSrc>>::type>::value,
                            "The source and the destination view are required to have the same element type!");

                        //-----------------------------------------------------------------------------
                        //! Constructor.
                        //-----------------------------------------------------------------------------
                        TaskCopy(
                            TViewDst & viewDst,
                            TViewSrc const & viewSrc,
                            TExtent const & extent) :
                                m_extentWidth(extent::getWidth(extent)),
                                m_extentWidthBytes(static_cast<Size>(m_extentWidth * sizeof(elem::Elem<TViewDst>))),
                                m_dstWidth(static_cast<Size>(extent::getWidth(viewDst))),
                                m_srcWidth(static_cast<Size>(extent::getWidth(viewSrc))),
                                m_dstBufWidth(static_cast<Size>(extent::getWidth(viewDst))),
                                m_srcBufWidth(static_cast<Size>(extent::getWidth(viewSrc))),

                                m_extentHeight(extent::getHeight(extent)),
                                m_dstHeight(static_cast<Size>(extent::getHeight(viewDst))),
                                m_srcHeight(static_cast<Size>(extent::getHeight(viewSrc))),
                                m_dstBufHeight(static_cast<Size>(extent::getHeight(viewDst))),
                                m_srcBufHeight(static_cast<Size>(extent::getHeight(viewSrc))),

                                m_extentDepth(extent::getDepth(extent)),
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                                m_dstDepth(static_cast<Size>(extent::getDepth(viewDst))),
                                m_srcDepth(static_cast<Size>(extent::getDepth(viewSrc))),
#endif
                                m_dstPitchBytesX(static_cast<Size>(mem::view::getPitchBytes<dim::Dim<TViewDst>::value - 1u>(viewDst))),
                                m_srcPitchBytesX(static_cast<Size>(mem::view::getPitchBytes<dim::Dim<TViewSrc>::value - 1u>(viewSrc))),
                                m_dstPitchBytesY(static_cast<Size>(mem::view::getPitchBytes<dim::Dim<TViewDst>::value - (2u % dim::Dim<TViewDst>::value)>(viewDst))),
                                m_srcPitchBytesY(static_cast<Size>(mem::view::getPitchBytes<dim::Dim<TViewSrc>::value - (2u % dim::Dim<TViewDst>::value)>(viewSrc))),                                

                                m_dstMemNative(reinterpret_cast<std::uint8_t *>(mem::view::getPtrNative(viewDst))),
                                m_srcMemNative(reinterpret_cast<std::uint8_t const *>(mem::view::getPtrNative(viewSrc)))
                        {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            assert(m_extentWidth <= m_dstWidth);
                            assert(m_extentHeight <= m_dstHeight);
                            assert(m_extentDepth <= m_dstDepth);
                            assert(m_extentWidth <= m_srcWidth);
                            assert(m_extentHeight <= m_srcHeight);
                            assert(m_extentDepth <= m_srcDepth);
                            assert(m_extentWidthBytes <= m_dstPitchBytesX);
                            assert(m_extentWidthBytes <= m_srcPitchBytesX);
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
                                << " dptr: " << reinterpret_cast<void *>(m_dstMemNative)
                                << " dpitchb: " << m_dstPitchBytesX
                                << " dbufw: " << m_dstBufWidth
                                << " dbufh: " << m_dstBufHeight
                                << " sw: " << m_srcWidth
                                << " sh: " << m_srcHeight
                                << " sd: " << m_srcDepth
                                << " sptr: " << reinterpret_cast<void const *>(m_srcMemNative)
                                << " spitchb: " << m_srcPitchBytesX
                                << " sbufw: " << m_srcBufWidth
                                << " sbufh: " << m_srcBufHeight
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


#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            printDebug();
#endif
                            auto const equalWidths(
                                (m_extentWidth == m_dstWidth)
                                && (m_extentWidth == m_srcWidth)
                                && (m_extentWidth == m_dstBufWidth)
                                && (m_extentWidth == m_srcBufWidth));

                            // If:
                            // - the copy extent width is identical to the dst and src extent width
                            // - the copy extent width is identical to the dst and src memory buffer extent width
                            // - the src and dst pitch is identical
                            // -> we can copy whole slices at once overwriting the pitch bytes
                            auto const copySliceAtOnce(
                                equalWidths
                                && (m_dstPitchBytesX == m_srcPitchBytesX));

                            // If:
                            // - the copy extent width and height are identical to the dst and src extent width and height
                            // - the copy extent width and height are identical to the dst and src memory buffer extent width and height
                            // - the src and dst slice size is identical
                            // -> we can copy the whole memory at once overwriting the pitch bytes
                            auto const copyAllAtOnce(
                                (m_extentHeight == m_dstHeight)
                                && (m_extentHeight == m_srcHeight)
                                && (m_extentHeight == m_dstBufHeight)
                                && (m_extentHeight == m_srcBufHeight)
                                && (m_dstPitchBytesY == m_srcPitchBytesY)
                                && copySliceAtOnce);

                            if(copyAllAtOnce)
                            {
                                std::memcpy(
                                    reinterpret_cast<void *>(m_dstMemNative),
                                    reinterpret_cast<void const *>(m_srcMemNative),
                                    m_dstPitchBytesX * m_extentHeight * m_extentDepth);
                            }
                            else
                            {
                                for(auto z(decltype(m_extentDepth)(0)); z < m_extentDepth; ++z)
                                {
                                    if(copySliceAtOnce)
                                    {
                                        std::memcpy(
                                            reinterpret_cast<void *>(m_dstMemNative + z*m_dstPitchBytesY),
                                            reinterpret_cast<void const *>(m_srcMemNative + z*m_srcPitchBytesY),
                                            m_dstPitchBytesX*m_extentHeight);
                                    }
                                    else
                                    {
                                        for(auto y((decltype(m_extentHeight)(0))); y < m_extentHeight; ++y)
                                        {
                                            std::memcpy(
                                                reinterpret_cast<void *>(m_dstMemNative + y*m_dstPitchBytesX + z*m_dstPitchBytesY),
                                                reinterpret_cast<void const *>(m_srcMemNative + y*m_srcPitchBytesX + z*m_srcPitchBytesY),
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
                        Size m_dstPitchBytesX;
                        Size m_srcPitchBytesX;
                        Size m_dstPitchBytesY;
                        Size m_srcPitchBytesY;                        

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
                        typename TExtent,
                        typename TViewSrc,
                        typename TViewDst>
                    ALPAKA_FN_HOST static auto taskCopy(
                        TViewDst & viewDst,
                        TViewSrc const & viewSrc,
                        TExtent const & extent)
                    -> cpu::detail::TaskCopy<
                        TViewDst,
                        TViewSrc,
                        TExtent>
                    {
                        return
                            cpu::detail::TaskCopy<
                                TViewDst,
                                TViewSrc,
                                TExtent>(
                                    viewDst,
                                    viewSrc,
                                    extent);
                    }
                };
            }
        }
    }
}
