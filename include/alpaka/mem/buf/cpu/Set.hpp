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
                        static_assert(
                            dim::Dim<TView>::value == dim::Dim<TExtent>::value,
                            "The destination view and the extent are required to have the same dimensionality!");

                        //-----------------------------------------------------------------------------
                        //! Constructor.
                        //-----------------------------------------------------------------------------
                        TaskSet(
                            TView & view,
                            std::uint8_t const & byte,
                            TExtent const & extent) :
                                m_view(view),
                                m_byte(byte),
                                m_extent(extent)
                        {}
                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST auto operator()() const
                        -> void
                        {
                            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                            auto const extentWidth(extent::getWidth(m_extent));
                            auto const extentHeight(extent::getHeight(m_extent));
                            auto const extentDepth(extent::getDepth(m_extent));
                            auto const dstWidth(extent::getWidth(m_view));
                            auto const dstHeight(extent::getHeight(m_view));
        #if (!defined(NDEBUG)) || (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                            auto const dstDepth(extent::getDepth(m_view));
        #endif
                            assert(extentWidth <= dstWidth);
                            assert(extentHeight <= dstHeight);
                            assert(extentDepth <= dstDepth);

                            auto const extentWidthBytes(extentWidth * sizeof(elem::Elem<TView>));
                            auto const dstPitchBytes(mem::view::getPitchBytes<dim::Dim<TView>::value - 1u>(m_view));
                            assert(extentWidthBytes <= dstPitchBytes);

                            auto const dstNativePtr(reinterpret_cast<std::uint8_t *>(mem::view::getPtrNative(m_view)));
                            auto const dstSliceSizeBytes(dstPitchBytes * dstHeight);

                            auto const dstBufWidth(extent::getWidth(m_view));
                            auto const dstBufHeight(extent::getHeight(m_view));

                            int iByte(static_cast<int>(m_byte));

        #if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            std::cout << BOOST_CURRENT_FUNCTION
                                << " ew: " << extentWidth
                                << " eh: " << extentHeight
                                << " ed: " << extentDepth
                                << " ewb: " << extentWidthBytes
                                << " dw: " << dstWidth
                                << " dh: " << dstHeight
                                << " dd: " << dstDepth
                                << " dptr: " << reinterpret_cast<void *>(dstNativePtr)
                                << " dpitchb: " << dstPitchBytes
                                << " dbufw: " << dstBufWidth
                                << " dbufh: " << dstBufHeight
                                << std::endl;
        #endif
                            // If:
                            // - the set extent width is identical to the dst extent width
                            // -> we can set whole slices at once overwriting the pitch bytes
                            auto const copySliceAtOnce(
                                (extentWidth == dstWidth)
                                && (extentWidth == dstBufWidth));

                            // If:
                            // - the set extent width and height are identical to the dst extent width and height
                            // -> we can set the whole memory at once overwriting the pitch bytes
                            auto const copyAllAtOnce(
                                copySliceAtOnce
                                && (extentHeight == dstHeight)
                                && (extentHeight == dstBufHeight));

                            if(copyAllAtOnce)
                            {
                                std::memset(
                                    reinterpret_cast<void *>(dstNativePtr),
                                    iByte,
                                    dstSliceSizeBytes*extentDepth);
                            }
                            else
                            {
                                for(auto z(decltype(extentDepth)(0)); z < extentDepth; ++z)
                                {
                                    if(copySliceAtOnce)
                                    {
                                        std::memset(
                                            reinterpret_cast<void *>(dstNativePtr + z*dstSliceSizeBytes),
                                            iByte,
                                            dstPitchBytes*extentHeight);
                                    }
                                    else
                                    {
                                        for(auto y(decltype(extentHeight)(0)); y < extentHeight; ++y)
                                        {
                                            std::memset(
                                                reinterpret_cast<void *>(dstNativePtr + y*dstPitchBytes + z*dstSliceSizeBytes),
                                                iByte,
                                                extentWidthBytes);
                                        }
                                    }
                                }
                            }
                        }

                        // FIXME: Copy view handle, do NOT take reference!
                        TView & m_view;
                        std::uint8_t const m_byte;
                        TExtent const m_extent;
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
