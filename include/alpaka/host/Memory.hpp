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

#include <alpaka/traits/Memory.hpp>         // traits::MemCopy, ...
#include <alpaka/traits/Extent.hpp>         // traits::getXXX

#include <alpaka/core/BasicDims.hpp>        // dim::Dim<N>
#include <alpaka/core/RuntimeExtents.hpp>   // extent::RuntimeExtents<TDim>

#include <alpaka/host/MemorySpace.hpp>      // MemSpaceHost

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
                public alpaka::extent::RuntimeExtents<TDim>
            {
            public:
                using Elem = TElem;
                using Dim = TDim;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor
                //-----------------------------------------------------------------------------
                template<
                    typename TExtent>
                MemBufHost(
                    TExtent const & extent):
                        RuntimeExtents<TDim>(extent),
                        m_spMem(
                            new TElem[computeElementCount(extent)],
                            [](TElem * pBuffer)
                            {
                                assert(pBuffer);
                                delete[] pBuffer;
                            })
                {}

            private:
                //-----------------------------------------------------------------------------
                //! \return The number of elements to allocate.
                //-----------------------------------------------------------------------------
                template<
                    typename TExtent>
                static std::size_t computeElementCount(
                    TExtent const & extent)
                {
                    auto const uiExtentElementCount(extent::getProductOfExtents(extent));
                    assert(uiExtentElementCount>0);

                    return uiExtentElementCount;
                }

            public:
                std::shared_ptr<TElem> m_spMem;
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
                typename std::enable_if<(TDim::value >= 1u) && (TDim::value <= 3u), void>::type
            >
            {
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
                typename std::enable_if<(TDim::value >= 2u) && (TDim::value <= 3u), void>::type
            >
            {
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
                typename std::enable_if<(TDim::value >= 3u) && (TDim::value <= 3u), void>::type
            >
            {
                static std::size_t getDepth(
                    host::detail::MemBufHost<TElem, TDim> const & extent)
                {
                    return extent.m_uiDepth;
                }
            };
        }

        namespace memory
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
                using type = alpaka::memory::MemSpaceHost;
            };

            //#############################################################################
            //! The MemBufHost memory element type get trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct GetMemElemType<
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
            struct GetMemBufType<
                TElem, TDim, alpaka::memory::MemSpaceHost>
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
                static TElem const * getNativePtr(
                    host::detail::MemBufHost<TElem, TDim> const & memBuf)
                {
                    return memBuf.m_spMem.get();
                }
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
                static std::size_t getPitchBytes(
                    host::detail::MemBufHost<TElem, TDim> const & memPitch)
                {
                    // No pitch on the host currently.
                    return alpaka::extent::getWidth(memPitch) * sizeof(TElem);
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
                alpaka::memory::MemSpaceHost
            >
            {
                template<
                    typename TExtent>
                static host::detail::MemBufHost<TElem, TDim> memAlloc(
                    TExtent const & extent)
                {
                    return host::detail::MemBufHost<TElem, TDim>(extent);
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
                alpaka::memory::MemSpaceHost,
                alpaka::memory::MemSpaceHost
            >
            {
                template<
                    typename TExtent, 
                    typename TMemBufSrc, 
                    typename TMemBufDst>
                static void memCopy(
                    TMemBufDst & memBufDst, 
                    TMemBufSrc const & memBufSrc, 
                    TExtent const & extent)
                {
                    static_assert(
                        std::is_same<alpaka::dim::GetDimT<TMemBufDst>, alpaka::dim::GetDimT<TMemBufSrc>>::value,
                        "The source and the destination buffers are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<alpaka::dim::GetDimT<TMemBufDst>, alpaka::dim::GetDimT<TExtent>>::value,
                        "The buffers and the extent are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<alpaka::memory::GetMemElemTypeT<TMemBufDst>, alpaka::memory::GetMemElemTypeT<TMemBufSrc>>::value,
                        "The source and the destination buffers are required to have the same element type!");

                    using Elem = alpaka::memory::GetMemElemTypeT<TMemBufDst>;

                    auto const uiExtentWidth(alpaka::extent::getWidth(extent));
                    auto const uiExtentHeight(alpaka::extent::getHeight(extent));
                    auto const uiExtentDepth(alpaka::extent::getDepth(extent));
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
                    auto const uiDstPitchBytes(alpaka::memory::getPitchBytes(memBufDst));
                    auto const uiSrcPitchBytes(alpaka::memory::getPitchBytes(memBufSrc));
                    assert(uiExtentWidthBytes <= uiDstPitchBytes);
                    assert(uiExtentWidthBytes <= uiSrcPitchBytes);

                    auto const pDstNative(reinterpret_cast<std::uint8_t *>(alpaka::memory::getNativePtr(memBufDst)));
                    auto const pSrcNative(reinterpret_cast<std::uint8_t const *>(alpaka::memory::getNativePtr(memBufSrc)));
                    auto const uiDstSliceSizeBytes(uiDstPitchBytes * uiDstHeight);
                    auto const uiSrcSliceSizeBytes(uiSrcPitchBytes * uiSrcHeight);

                    // If:
                    // - the copy extent width and height are identical to the dst and src extent width and height
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
                            // - the copy extent width is identical to the dst and src extent width
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
            };

            //#############################################################################
            //! The host accelerators memory set trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct MemSet<
                TDim, 
                alpaka::memory::MemSpaceHost
            >
            {
                template<
                    typename TExtent, 
                    typename TMemBuf>
                static void memSet(
                    TMemBuf & memBuf, 
                    std::uint8_t const & byte, 
                    TExtent const & extent)
                {
                    using Elem = alpaka::memory::GetMemElemTypeT<TMemBuf>;

                    static_assert(
                        std::is_same<alpaka::dim::GetDimT<TMemBuf>, alpaka::dim::GetDimT<TExtent>>::value,
                        "The destination buffer and the extent are required to have the same dimensionality!");

                    auto const uiExtentWidth(alpaka::extent::getWidth(extent));
                    auto const uiExtentHeight(alpaka::extent::getHeight(extent));
                    auto const uiExtentDepth(alpaka::extent::getDepth(extent));
                    auto const uiDstWidth(alpaka::extent::getWidth(memBuf));
                    auto const uiDstHeight(alpaka::extent::getHeight(memBuf));
                    auto const uiDstDepth(alpaka::extent::getDepth(memBuf));
                    assert(uiExtentWidth <= uiDstWidth);
                    assert(uiExtentHeight <= uiDstHeight);
                    assert(uiExtentDepth <= uiDstDepth);

                    auto const uiExtentWidthBytes(uiExtentWidth * sizeof(Elem));
                    auto const uiDstPitchBytes(alpaka::memory::getPitchBytes(memBuf));
                    assert(uiExtentWidthBytes <= uiDstPitchBytes);

                    auto const pDstNative(reinterpret_cast<std::uint8_t *>(alpaka::memory::getNativePtr(memBuf)));
                    auto const uiDstSliceSizeBytes(uiDstPitchBytes * uiDstHeight);

                    int iByte(static_cast<int>(byte));

                    // If:
                    // - the set extent width and height are identical to the dst extent width and height
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
                            // - the set extent width is identical to the dst extent width
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
            };
        }
    }
}
