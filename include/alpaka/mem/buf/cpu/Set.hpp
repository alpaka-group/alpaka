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
#include <alpaka/extent/Traits.hpp>         // view::getXXX
#include <alpaka/mem/view/Traits.hpp>       // view::Set, ...
#include <alpaka/stream/StreamCpuAsync.hpp> // stream::StreamCpuAsync
#include <alpaka/stream/StreamCpuSync.hpp>  // stream::StreamCpuSync

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
                        typename TBuf,
                        typename TExtents>
                    struct TaskSet
                    {
                        static_assert(
                            dim::Dim<TBuf>::value == dim::Dim<TExtents>::value,
                            "The destination buffer and the extents are required to have the same dimensionality!");

                        //-----------------------------------------------------------------------------
                        //! Constructor.
                        //-----------------------------------------------------------------------------
                        TaskSet(
                            TBuf & buf,
                            std::uint8_t const & byte,
                            TExtents const & extents) :
                                m_buf(buf),
                                m_byte(byte),
                                m_extents(extents)
                        {}
                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST auto operator()() const
                        -> void
                        {
                            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                            auto const extentWidth(extent::getWidth(m_extents));
                            auto const extentHeight(extent::getHeight(m_extents));
                            auto const extentDepth(extent::getDepth(m_extents));
                            auto const dstWidth(extent::getWidth(m_buf));
                            auto const dstHeight(extent::getHeight(m_buf));
        #if (!defined(NDEBUG)) || (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                            auto const dstDepth(extent::getDepth(m_buf));
        #endif
                            assert(extentWidth <= dstWidth);
                            assert(extentHeight <= dstHeight);
                            assert(extentDepth <= dstDepth);

                            auto const extentWidthBytes(extentWidth * sizeof(mem::view::Elem<TBuf>));
                            auto const dstPitchBytes(mem::view::getPitchBytes<dim::Dim<TBuf>::value - 1u>(m_buf));
                            assert(extentWidthBytes <= dstPitchBytes);

                            auto const dstNativePtr(reinterpret_cast<std::uint8_t *>(mem::view::getPtrNative(m_buf)));
                            auto const dstSliceSizeBytes(dstPitchBytes * dstHeight);

                            auto const & dstBuf(mem::view::getBuf(m_buf));
                            auto const dstBufWidth(extent::getWidth(dstBuf));
                            auto const dstBufHeight(extent::getHeight(dstBuf));

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
                                << " dbasew: " << dstBufWidth
                                << " dbaseh: " << dstBufHeight
                                << std::endl;
        #endif
                            // If:
                            // - the set extents width is identical to the dst extents width
                            // -> we can set whole slices at once overwriting the pitch bytes
                            auto const copySliceAtOnce(
                                (extentWidth == dstWidth)
                                && (extentWidth == dstBufWidth));

                            // If:
                            // - the set extents width and height are identical to the dst extents width and height
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

                        // FIXME: Copy buffer handle, do NOT take reference!
                        TBuf & m_buf;
                        std::uint8_t const m_byte;
                        TExtents const m_extents;
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
                        typename TExtents,
                        typename TBuf>
                    ALPAKA_FN_HOST static auto taskSet(
                        TBuf & buf,
                        std::uint8_t const & byte,
                        TExtents const & extents)
                    -> cpu::detail::TaskSet<
                        TBuf,
                        TExtents>
                    {
                        return
                            cpu::detail::TaskSet<
                                TBuf,
                                TExtents>(
                                    buf,
                                    byte,
                                    extents);
                    }
                };
            }
        }
    }
}
