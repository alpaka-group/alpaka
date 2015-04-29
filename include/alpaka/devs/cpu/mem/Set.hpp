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

#include <alpaka/core/BasicDims.hpp>    // dim::Dim<N>

#include <alpaka/traits/mem/Buf.hpp>    // traits::Alloc, ...
#include <alpaka/traits/Extent.hpp>     // traits::getXXX

#include <boost/core/ignore_unused.hpp> // boost::ignore_unused

#include <cassert>                      // assert
#include <cstring>                      // std::memset

namespace alpaka
{
    namespace devs
    {
        namespace cpu
        {
            namespace detail
            {
                class DevCpu;
                class StreamCpu;
            }
        }
    }
}

namespace alpaka
{
    namespace traits
    {
        namespace mem
        {
            //#############################################################################
            //! The cpu device memory set trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct Set<
                TDim,
                devs::cpu::detail::DevCpu>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                template<
                    typename TBuf,
                    typename TExtents>
                ALPAKA_FCT_HOST static auto set(
                    TBuf & bufDst,
                    std::uint8_t const & byte,
                    TExtents const & extents)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    using Elem = alpaka::mem::ElemT<TBuf>;

                    static_assert(
                        alpaka::dim::DimT<TBuf>::value == alpaka::dim::DimT<TExtents>::value,
                        "The destination buffer and the extents are required to have the same dimensionality!");

                    auto const uiExtentWidth(alpaka::extent::getWidth<UInt>(extents));
                    auto const uiExtentHeight(alpaka::extent::getHeight<UInt>(extents));
                    auto const uiExtentDepth(alpaka::extent::getDepth<UInt>(extents));
                    auto const uiDstWidth(alpaka::extent::getWidth<UInt>(bufDst));
                    auto const uiDstHeight(alpaka::extent::getHeight<UInt>(bufDst));
#if (!defined(NDEBUG)) || (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                    auto const uiDstDepth(alpaka::extent::getDepth<UInt>(bufDst));
#endif
                    assert(uiExtentWidth <= uiDstWidth);
                    assert(uiExtentHeight <= uiDstHeight);
                    assert(uiExtentDepth <= uiDstDepth);

                    auto const uiExtentWidthBytes(uiExtentWidth * sizeof(Elem));
                    auto const uiDstPitchBytes(alpaka::mem::getPitchBytes<0u, UInt>(bufDst));
                    assert(uiExtentWidthBytes <= uiDstPitchBytes);

                    auto const pDstNative(reinterpret_cast<std::uint8_t *>(alpaka::mem::getPtrNative(bufDst)));
                    auto const uiDstSliceSizeBytes(uiDstPitchBytes * uiDstHeight);

                    auto const & dstBuf(alpaka::mem::getBase(bufDst));
                    auto const uiDstBufWidth(alpaka::extent::getWidth<UInt>(dstBuf));
                    auto const uiDstBufHeight(alpaka::extent::getHeight<UInt>(dstBuf));

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
                        for(UInt z(0); z < uiExtentDepth; ++z)
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
                                for(UInt y(0); y < uiExtentHeight; ++y)
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
                // \TODO: Implement asynchronous cpu set.
                //-----------------------------------------------------------------------------
                template<
                    typename TBuf,
                    typename TExtents,
                    typename TDev>
                ALPAKA_FCT_HOST static auto set(
                    TBuf & bufDst,
                    std::uint8_t const & byte,
                    TExtents const & extents,
                    devs::cpu::detail::StreamCpu const & stream)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    
                    boost::ignore_unused(stream);

                    copy(
                        bufDst,
                        byte,
                        extents);
                }
            };
        }
    }
}
