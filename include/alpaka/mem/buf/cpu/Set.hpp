            /**
* \file
* Copyright 2014-2015 Benjamin Worpitz, Erik Zenker
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
#include <alpaka/extent/Traits.hpp>         // mem::view::getXXX
#include <alpaka/mem/view/Traits.hpp>       // mem::view::Set, ...

#include <cassert>                          // assert
#include <cstring>                          // std::memset

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
                    //! The CPU device memory set task.
                    //!
                    //! Set CPU memory.
                    //#############################################################################
                    template<
                        typename TView,
                        typename TExtent>
                    struct TaskSet
                    {
                        using Size = size::Size<TExtent>;

                        static_assert(
                            dim::Dim<TView>::value == dim::Dim<TExtent>::value,
                            "The destination view and the extent are required to have the same dimensionality!");

                        static_assert(
                            dim::Dim<TView>::value <= 3u,
                            "TaskSet for DevCpu does not currently support views with more than 3 dimensions!");

                        //-----------------------------------------------------------------------------
                        //! Constructor.
                        //-----------------------------------------------------------------------------
                        TaskSet(
                            TView & view,
                            std::uint8_t const & byte,
                            TExtent const & extent) :
                                m_byte(byte),
                                m_extentWidth(static_cast<Size>(extent::getWidth(extent))),
                                m_extentHeight(static_cast<Size>(extent::getHeight(extent))),
                                m_extentDepth(static_cast<Size>(extent::getDepth(extent))),
#if (!defined(NDEBUG)) || (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                                m_dstWidth(static_cast<Size>(extent::getWidth(view))),
                                m_dstHeight(static_cast<Size>(extent::getHeight(view))),
                                m_dstDepth(static_cast<Size>(extent::getDepth(view))),
#endif
                                m_extentWidthBytes(m_extentWidth * static_cast<Size>(sizeof(elem::Elem<TView>))),
                                m_dstPitchBytesX(mem::view::getPitchBytes<dim::Dim<TView>::value - 1u>(view)),
                                m_dstPitchBytesY(mem::view::getPitchBytes<dim::Dim<TView>::value - (2u % dim::Dim<TView>::value)>(view)),
                                m_dstNativePtr(reinterpret_cast<std::uint8_t *>(mem::view::getPtrNative(view))),
                                m_dstBufWidth(static_cast<Size>(extent::getWidth(view))),
                                m_dstBufHeight(static_cast<Size>(extent::getHeight(view)))
                        {
                            assert(m_extentWidth <= m_dstWidth);
                            assert(m_extentHeight <= m_dstHeight);
                            assert(m_extentDepth <= m_dstDepth);
                            assert(m_extentWidthBytes <= m_dstPitchBytesX);
                        }
                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST auto operator()() const
                        -> void
                        {
                            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            std::cout << BOOST_CURRENT_FUNCTION
                                << " ew: " << m_extentWidth
                                << " eh: " << m_extentHeight
                                << " ed: " << m_extentDepth
                                << " ewb: " << m_extentWidthBytes
                                << " dw: " << m_dstWidth
                                << " dh: " << m_dstHeight
                                << " dd: " << m_dstDepth
                                << " dptr: " << reinterpret_cast<void *>(m_dstNativePtr)
                                << " dpitchbX: " << m_dstPitchBytesX
                                << " dpitchbY: " << m_dstPitchBytesY
                                << " dbufw: " << m_dstBufWidth
                                << " dbufh: " << m_dstBufHeight
                                << std::endl;
#endif

                            for(Size z = static_cast<Size>(0); z < m_extentDepth; ++z)
                            {
                                for(Size y = static_cast<Size>(0); y < m_extentHeight; ++y)
                                {
                                    std::memset(
                                        reinterpret_cast<void *>(m_dstNativePtr + y*m_dstPitchBytesX + z*m_dstPitchBytesY),
                                        m_byte,
                                        static_cast<std::size_t>(m_extentWidthBytes));
                                }
                            }
                        }

                        std::uint8_t const m_byte;
                        Size const m_extentWidth;
                        Size const m_extentHeight;
                        Size const m_extentDepth;
#if (!defined(NDEBUG)) || (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                        Size const m_dstWidth;
                        Size const m_dstHeight;
                        Size const m_dstDepth;
#endif
                        Size const m_extentWidthBytes;
                        Size const m_dstPitchBytesX;
                        Size const m_dstPitchBytesY;
                        std::uint8_t * const m_dstNativePtr;
                        Size const m_dstBufWidth;
                        Size const m_dstBufHeight;

                    };
                }
            }

            namespace traits
            {
                //#############################################################################
                //! The CPU device memory set trait specialization.
                //#############################################################################
                template<
                    typename TDim>
                struct TaskSet<
                    TDim,
                    dev::DevCpu>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent,
                        typename TView>
                    ALPAKA_FN_HOST static auto taskSet(
                        TView & view,
                        std::uint8_t const & byte,
                        TExtent const & extent)
                    -> cpu::detail::TaskSet<
                        TView,
                        TExtent>
                    {
                        return
                            cpu::detail::TaskSet<
                                TView,
                                TExtent>(
                                    view,
                                    byte,
                                    extent);
                    }
                };
            }
        }
    }
}
