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
#include <alpaka/stream/StreamCpuAsync.hpp> // StreamCpuAsync

#include <boost/core/ignore_unused.hpp>     // boost::ignore_unused

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
            namespace traits
            {
                //#############################################################################
                //! The CPU device memory set trait specialization.
                //#############################################################################
                template<
                    typename TDim>
                struct Set<
                    TDim,
                    dev::DevCpu>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TBuf,
                        typename TExtents>
                    ALPAKA_FN_HOST static auto set(
                        TBuf & buf,
                        std::uint8_t const & byte,
                        TExtents const & extents)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        using Elem = mem::view::Elem<TBuf>;

                        static_assert(
                            dim::Dim<TBuf>::value == dim::Dim<TExtents>::value,
                            "The destination buffer and the extents are required to have the same dimensionality!");

                        auto const uiExtentWidth(extent::getWidth(extents));
                        auto const uiExtentHeight(extent::getHeight(extents));
                        auto const uiExtentDepth(extent::getDepth(extents));
                        auto const uiDstWidth(extent::getWidth(buf));
                        auto const uiDstHeight(extent::getHeight(buf));
#if (!defined(NDEBUG)) || (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                        auto const uiDstDepth(extent::getDepth(buf));
#endif
                        assert(uiExtentWidth <= uiDstWidth);
                        assert(uiExtentHeight <= uiDstHeight);
                        assert(uiExtentDepth <= uiDstDepth);

                        auto const uiExtentWidthBytes(uiExtentWidth * sizeof(Elem));
                        auto const uiDstPitchBytes(mem::view::getPitchBytes<dim::Dim<TBuf>::value - 1u>(buf));
                        assert(uiExtentWidthBytes <= uiDstPitchBytes);

                        auto const pDstNative(reinterpret_cast<std::uint8_t *>(mem::view::getPtrNative(buf)));
                        auto const uiDstSliceSizeBytes(uiDstPitchBytes * uiDstHeight);

                        auto const & dstBuf(mem::view::getBuf(buf));
                        auto const uiDstBufWidth(extent::getWidth(dstBuf));
                        auto const uiDstBufHeight(extent::getHeight(dstBuf));

                        int iByte(static_cast<int>(byte));

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
                            << std::endl;
#endif
                        // If:
                        // - the set extents width and height are identical to the dst extents width and height
                        // -> we can set the whole memory at once overwriting the pitch bytes
                        if((uiExtentWidth == uiDstWidth)
                            && (uiExtentHeight == uiDstHeight)
                            && (uiExtentWidth == uiDstBufWidth)
                            && (uiExtentHeight == uiDstBufHeight))
                        {
                            std::memset(
                                reinterpret_cast<void *>(pDstNative),
                                iByte,
                                uiDstSliceSizeBytes*uiExtentDepth);
                        }
                        else
                        {
                            for(auto z(decltype(uiExtentDepth)(0)); z < uiExtentDepth; ++z)
                            {
                                // If:
                                // - the set extents width is identical to the dst extents width
                                // -> we can set whole slices at once overwriting the pitch bytes
                                if((uiExtentWidth == uiDstWidth)
                                    && (uiExtentWidth == uiDstBufWidth))
                                {
                                    std::memset(
                                        reinterpret_cast<void *>(pDstNative + z*uiDstSliceSizeBytes),
                                        iByte,
                                        uiDstPitchBytes*uiExtentHeight);
                                }
                                else
                                {
                                    for(auto y(decltype(uiExtentHeight)(0)); y < uiExtentHeight; ++y)
                                    {
                                        std::memset(
                                            reinterpret_cast<void *>(pDstNative + y*uiDstPitchBytes + z*uiDstSliceSizeBytes),
                                            iByte,
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
                        typename TBuf,
                        typename TExtents,
                        typename TDev>
                    ALPAKA_FN_HOST static auto set(
                        TBuf & buf,
                        std::uint8_t const & byte,
                        TExtents const & extents,
                        stream::StreamCpuAsync const & stream)
                    -> void
                    {
                        boost::ignore_unused(stream);

                        stream.m_spAsyncStreamCpu->m_workerThread.enqueueTask(
                            [&buf, &byte, extents]()
                            {
                                set(
                                    buf,
                                    byte,
                                    extents);
                            });
                    }
                };
            }
        }
    }
}
