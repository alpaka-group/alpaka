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

#include <alpaka/traits/Mem.hpp>            // traits::MemCopy, ...
#include <alpaka/traits/Extents.hpp>        // traits::getXXX

#include <alpaka/core/BasicDims.hpp>        // dim::Dim<N>
#include <alpaka/core/BasicExtents.hpp>     // extent::BasicExtents<TDim>

#include <alpaka/host/MemSpace.hpp>         // MemSpaceHost
#include <alpaka/host/Stream.hpp>           // StreamHost

#include <cstring>                          // std::memcpy, std::memset
#include <cassert>                          // assert
#include <cstdint>                          // std::size_t
#include <memory>                           // std::shared_ptr

namespace alpaka
{
    namespace host
    {
        namespace detail
        {
            //#############################################################################
            //! The host memory buffer.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            class MemBufHost :
                public extent::BasicExtents<TDim>
            {
            private:
                using Extent = extent::BasicExtents<TDim>;
                using Elem = TElem;
                using Dim = TDim;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents>
                MemBufHost(
                    TExtents const & extents) :
                        Extent(extents),
                        m_spMem(new TElem[computeElementCount(extents)], &MemBufHost::freeBuffer),
                        m_uiPitchBytes(extent::getWidth(extents) * sizeof(TElem))
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                }

            private:
                //-----------------------------------------------------------------------------
                //! \return The number of elements to allocate.
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents>
                static std::size_t computeElementCount(
                    TExtents const & extents)
                {
                    auto const uiExtentsElementCount(extent::getProductOfExtents(extents));
                    assert(uiExtentsElementCount>0);

                    return uiExtentsElementCount;
                }
                //-----------------------------------------------------------------------------
                //! Frees the shared buffer.
                //-----------------------------------------------------------------------------
                static void freeBuffer(
                    TElem * pBuffer)
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    assert(pBuffer);
                    delete[] pBuffer;
                }

            public:
                std::shared_ptr<TElem> m_spMem;

                std::size_t m_uiPitchBytes;
            };
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for host::detail::MemBufHost.
    //-----------------------------------------------------------------------------
    namespace traits
    {
        namespace dim
        {
            //#############################################################################
            //! The MemBufHost dimension getter trait.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct GetDim<
                host::detail::MemBufHost<TElem, TDim>>
            {
                using type = TDim;
            };
        }

        namespace extent
        {
            //#############################################################################
            //! The MemBufHost width get trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct GetWidth<
                host::detail::MemBufHost<TElem, TDim>,
                typename std::enable_if<(TDim::value >= 1u) && (TDim::value <= 3u)>::type>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                static std::size_t getWidth(
                    host::detail::MemBufHost<TElem, TDim> const & extent)
                {
                    return extent.m_uiWidth;
                }
            };

            //#############################################################################
            //! The MemBufHost height get trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct GetHeight<
                host::detail::MemBufHost<TElem, TDim>,
                typename std::enable_if<(TDim::value >= 2u) && (TDim::value <= 3u)>::type>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                static std::size_t getHeight(
                    host::detail::MemBufHost<TElem, TDim> const & extent)
                {
                    return extent.m_uiHeight;
                }
            };
            //#############################################################################
            //! The MemBufHost depth get trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct GetDepth<
                host::detail::MemBufHost<TElem, TDim>,
                typename std::enable_if<(TDim::value >= 3u) && (TDim::value <= 3u)>::type>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                static std::size_t getDepth(
                    host::detail::MemBufHost<TElem, TDim> const & extent)
                {
                    return extent.m_uiDepth;
                }
            };
        }

        namespace mem
        {
            //#############################################################################
            //! The MemBufHost memory space trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct GetMemSpace<
                host::detail::MemBufHost<TElem, TDim>>
            {
                using type = alpaka::mem::MemSpaceHost;
            };

            //#############################################################################
            //! The MemBufHost memory element type get trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct GetMemElem<
                host::detail::MemBufHost<TElem, TDim>>
            {
                using type = TElem;
            };

            //#############################################################################
            //! The MemBufHost memory buffer type trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct GetMemBuf<
                TElem, TDim, alpaka::mem::MemSpaceHost>
            {
                using type = host::detail::MemBufHost<TElem, TDim>;
            };

            //#############################################################################
            //! The MemBufHost native pointer get trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct GetNativePtr<
                host::detail::MemBufHost<TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                static TElem const * getNativePtr(
                    host::detail::MemBufHost<TElem, TDim> const & memBuf)
                {
                    return memBuf.m_spMem.get();
                }
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                static TElem * getNativePtr(
                    host::detail::MemBufHost<TElem, TDim> & memBuf)
                {
                    return memBuf.m_spMem.get();
                }
            };

            //#############################################################################
            //! The MemBufHost pitch get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct GetPitchBytes<
                host::detail::MemBufHost<TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                static std::size_t getPitchBytes(
                    host::detail::MemBufHost<TElem, TDim> const & memPitch)
                {
                    // No pitch on the host currently.
                    return memPitch.m_uiPitchBytes;
                }
            };

            //#############################################################################
            //! The host accelerators memory allocation trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct MemAlloc<
                TElem,
                TDim,
                alpaka::mem::MemSpaceHost>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents>
                static host::detail::MemBufHost<TElem, TDim> memAlloc(
                    TExtents const & extents)
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    return host::detail::MemBufHost<TElem, TDim>(extents);
                }
            };

            //#############################################################################
            //! The host accelerators memory copy trait specialization.
            //!
            //! Copies from host memory into host memory.
            //#############################################################################
            template<
                typename TDim>
            struct MemCopy<
                TDim, 
                alpaka::mem::MemSpaceHost,
                alpaka::mem::MemSpaceHost>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents, 
                    typename TMemBufSrc, 
                    typename TMemBufDst>
                static void memCopy(
                    TMemBufDst & memBufDst, 
                    TMemBufSrc const & memBufSrc, 
                    TExtents const & extents)
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    static_assert(
                        std::is_same<alpaka::dim::GetDimT<TMemBufDst>, alpaka::dim::GetDimT<TMemBufSrc>>::value,
                        "The source and the destination buffers are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<alpaka::dim::GetDimT<TMemBufDst>, alpaka::dim::GetDimT<TExtents>>::value,
                        "The buffers and the extents are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<alpaka::mem::GetMemElemT<TMemBufDst>, alpaka::mem::GetMemElemT<TMemBufSrc>>::value,
                        "The source and the destination buffers are required to have the same element type!");

                    using Elem = alpaka::mem::GetMemElemT<TMemBufDst>;

                    auto const uiExtentWidth(alpaka::extent::getWidth(extents));
                    auto const uiExtentHeight(alpaka::extent::getHeight(extents));
                    auto const uiExtentDepth(alpaka::extent::getDepth(extents));
                    auto const uiDstWidth(alpaka::extent::getWidth(memBufDst));
                    auto const uiDstHeight(alpaka::extent::getHeight(memBufDst));
                    auto const uiDstDepth(alpaka::extent::getDepth(memBufDst));
                    auto const uiSrcWidth(alpaka::extent::getWidth(memBufSrc));
                    auto const uiSrcHeight(alpaka::extent::getHeight(memBufSrc));
                    auto const uiSrcDepth(alpaka::extent::getDepth(memBufSrc));
                    assert(uiExtentWidth <= uiDstWidth);
                    assert(uiExtentHeight <= uiDstHeight);
                    assert(uiExtentDepth <= uiDstDepth);
                    assert(uiExtentWidth <= uiSrcWidth);
                    assert(uiExtentHeight <= uiSrcHeight);
                    assert(uiExtentDepth <= uiSrcDepth);

                    auto const uiExtentWidthBytes(uiExtentWidth * sizeof(Elem));
                    auto const uiDstPitchBytes(alpaka::mem::getPitchBytes(memBufDst));
                    auto const uiSrcPitchBytes(alpaka::mem::getPitchBytes(memBufSrc));
                    assert(uiExtentWidthBytes <= uiDstPitchBytes);
                    assert(uiExtentWidthBytes <= uiSrcPitchBytes);

                    auto const pDstNative(reinterpret_cast<std::uint8_t *>(alpaka::mem::getNativePtr(memBufDst)));
                    auto const pSrcNative(reinterpret_cast<std::uint8_t const *>(alpaka::mem::getNativePtr(memBufSrc)));
                    auto const uiDstSliceSizeBytes(uiDstPitchBytes * uiDstHeight);
                    auto const uiSrcSliceSizeBytes(uiSrcPitchBytes * uiSrcHeight);

                    // If:
                    // - the copy extents width and height are identical to the dst and src extents width and height
                    // - the src and dst slice size is identical 
                    // -> we can copy the whole memory at once overwriting the pitch bytes
                    if((uiExtentWidth == uiDstWidth)
                        && (uiExtentWidth == uiSrcWidth)
                        && (uiExtentHeight == uiDstHeight)
                        && (uiExtentHeight == uiSrcHeight)
                        && (uiDstSliceSizeBytes == uiSrcSliceSizeBytes))
                    {
                        std::memcpy(
                            reinterpret_cast<void *>(pDstNative),
                            reinterpret_cast<void const *>(pSrcNative),
                            uiDstSliceSizeBytes*uiExtentDepth);
                    }
                    else
                    {
                        for(std::size_t z(0); z < uiExtentDepth; ++z)
                        {
                            // If:
                            // - the copy extents width is identical to the dst and src extents width
                            // - the src and dst pitch is identical 
                            // -> we can copy whole slices at once overwriting the pitch bytes
                            if((uiExtentWidth == uiDstWidth)
                                && (uiExtentWidth == uiSrcWidth)
                                && (uiDstPitchBytes == uiSrcPitchBytes))
                            {
                                std::memcpy(
                                    reinterpret_cast<void *>(pDstNative + z*uiDstSliceSizeBytes),
                                    reinterpret_cast<void const *>(pSrcNative + z*uiSrcSliceSizeBytes),
                                    uiDstPitchBytes*uiExtentHeight);
                            }
                            else
                            {
                                for(std::size_t y(0); y < uiExtentHeight; ++y)
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
                    typename TMemBufSrc, 
                    typename TMemBufDst>
                static void memCopy(
                    TMemBufDst & memBufDst, 
                    TMemBufSrc const & memBufSrc, 
                    TExtents const & extents,
                    host::detail::StreamHost const &)
                {
                    // \TODO: Implement asynchronous host memCopy.
                    memCopy(
                        memBufDst,
                        memBufSrc,
                        extents);
                }
            };

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
                static void memSet(
                    TMemBuf & memBuf, 
                    std::uint8_t const & byte, 
                    TExtents const & extents)
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    using Elem = alpaka::mem::GetMemElemT<TMemBuf>;

                    static_assert(
                        std::is_same<alpaka::dim::GetDimT<TMemBuf>, alpaka::dim::GetDimT<TExtents>>::value,
                        "The destination buffer and the extents are required to have the same dimensionality!");

                    auto const uiExtentWidth(alpaka::extent::getWidth(extents));
                    auto const uiExtentHeight(alpaka::extent::getHeight(extents));
                    auto const uiExtentDepth(alpaka::extent::getDepth(extents));
                    auto const uiDstWidth(alpaka::extent::getWidth(memBuf));
                    auto const uiDstHeight(alpaka::extent::getHeight(memBuf));
                    auto const uiDstDepth(alpaka::extent::getDepth(memBuf));
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
                static void memSet(
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
