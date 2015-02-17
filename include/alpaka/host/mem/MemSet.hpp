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

#include <alpaka/host/MemSpace.hpp>         // MemSpaceHost
#include <alpaka/host/Stream.hpp>           // StreamHost

#include <alpaka/core/BasicDims.hpp>        // dim::Dim<N>

#include <alpaka/traits/mem/MemBufBase.hpp> // traits::MemAlloc, ...
#include <alpaka/traits/Extents.hpp>        // traits::getXXX

#include <cassert>                          // assert
#include <cstddef>                          // std::size_t
#include <cstring>                          // std::memset

namespace alpaka
{
    namespace traits
    {
        namespace mem
        {
            //#############################################################################
            //! The host accelerators memory set trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct MemSet<
                TDim, 
                alpaka::mem::MemSpaceHost>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                template<
                    typename TMemBuf, 
                    typename TExtents>
                ALPAKA_FCT_HOST static void memSet(
                    TMemBuf & memBuf, 
                    std::uint8_t const & byte, 
                    TExtents const & extents)
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    using Elem = alpaka::mem::MemElemT<TMemBuf>;

                    static_assert(
                        std::is_same<alpaka::dim::DimT<TMemBuf>, alpaka::dim::DimT<TExtents>>::value,
                        "The destination buffer and the extents are required to have the same dimensionality!");

                    auto const uiExtentWidth(alpaka::extent::getWidth(extents));
                    auto const uiExtentHeight(alpaka::extent::getHeight(extents));
                    auto const uiExtentDepth(alpaka::extent::getDepth(extents));
                    auto const uiDstWidth(alpaka::extent::getWidth(memBuf));
                    auto const uiDstHeight(alpaka::extent::getHeight(memBuf));
#ifndef NDEBUG
                    auto const uiDstDepth(alpaka::extent::getDepth(memBuf));
#endif
                    assert(uiExtentWidth <= uiDstWidth);
                    assert(uiExtentHeight <= uiDstHeight);
                    assert(uiExtentDepth <= uiDstDepth);
                       
                    auto const uiExtentWidthBytes(uiExtentWidth * sizeof(Elem));
                    auto const uiDstPitchBytes(alpaka::mem::getPitchBytes(memBuf));
                    assert(uiExtentWidthBytes <= uiDstPitchBytes);

                    auto const pDstNative(reinterpret_cast<std::uint8_t *>(alpaka::mem::getNativePtr(memBuf)));
                    auto const uiDstSliceSizeBytes(uiDstPitchBytes * uiDstHeight);

                    int iByte(static_cast<int>(byte));

                    // If:
                    // - the set extents width and height are identical to the dst extents width and height
                    // -> we can set the whole memory at once overwriting the pitch bytes
                    if((uiExtentWidth == uiDstWidth)
                        && (uiExtentHeight == uiDstHeight))
                    {
                        std::memset(
                            reinterpret_cast<void *>(pDstNative),
                            iByte,
                            uiDstSliceSizeBytes*uiExtentDepth);
                    }
                    else
                    {
                        for(std::size_t z(0); z < uiExtentDepth; ++z)
                        {
                            // If: 
                            // - the set extents width is identical to the dst extents width
                            // -> we can set whole slices at once overwriting the pitch bytes
                            if(uiExtentWidth == uiDstWidth)
                            {
                                std::memset(
                                    reinterpret_cast<void *>(pDstNative + z*uiDstSliceSizeBytes),
                                    iByte,
                                    uiDstPitchBytes*uiExtentHeight);
                            }
                            else
                            {
                                for(std::size_t y(0); y < uiExtentHeight; ++y)
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
                    typename TMemBuf, 
                    typename TExtents,
                    typename TStream>
                ALPAKA_FCT_HOST static void memSet(
                    TMemBuf & memBuf, 
                    std::uint8_t const & byte, 
                    TExtents const & extents,
                    host::detail::StreamHost const &)
                {
                    // \TODO: Implement asynchronous host memSet.
                    memSet(
                        memBuf,
                        byte,
                        extents);
                }
            };
        }
    }
}
