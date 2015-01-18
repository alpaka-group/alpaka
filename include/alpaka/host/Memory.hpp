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

#include <alpaka/host/MemorySpace.hpp>      // MemorySpaceHost

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
                typename TElement,
                typename TDim>
            class MemBufHost :
                public alpaka::extent::RuntimeExtents<TDim>
            {
            public:
                using Element = TElement;
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
                        new TElement[computeElementCount(extent)],
                        [](TElement * pBuffer)
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
                std::shared_ptr<TElement> m_spMem;
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
                typename TElement, 
                typename TDim>
            struct GetDim<host::detail::MemBufHost<TElement, TDim>>
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
                typename TElement, 
                typename TDim>
            struct GetWidth<
                host::detail::MemBufHost<TElement, TDim>,
                typename std::enable_if<(TDim::value >= 1) && (TDim::value <= 3), void>::type
            >
            {
                static std::size_t getWidth(
                    host::detail::MemBufHost<TElement, TDim> const & extent)
                {
                    return extent.m_uiWidth;
                }
            };

            //#############################################################################
            //! The MemBufHost height get trait specialization.
            //#############################################################################
            template<
                typename TElement, 
                typename TDim>
            struct GetHeight<
                host::detail::MemBufHost<TElement, TDim>,
                typename std::enable_if<(TDim::value >= 2) && (TDim::value <= 3), void>::type
            >
            {
                static std::size_t getHeight(
                    host::detail::MemBufHost<TElement, TDim> const & extent)
                {
                    return extent.m_uiHeight;
                }
            };
            //#############################################################################
            //! The MemBufHost depth get trait specialization.
            //#############################################################################
            template<
                typename TElement, 
                typename TDim>
            struct GetDepth<
                host::detail::MemBufHost<TElement, TDim>,
                typename std::enable_if<(TDim::value >= 3) && (TDim::value <= 3), void>::type
            >
            {
                static std::size_t getDepth(
                    host::detail::MemBufHost<TElement, TDim> const & extent)
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
                typename TElement, 
                typename TDim>
            struct GetMemSpace<
                host::detail::MemBufHost<TElement, TDim>>
            {
                using type = MemorySpaceHost;
            };

            //#############################################################################
            //! The MemBufHost memory element type get trait specialization.
            //#############################################################################
            template<
                typename TElement, 
                typename TDim>
            struct GetMemElemType<
                host::detail::MemBufHost<TElement, TDim>>
            {
                using type = TElement;
            };

            //#############################################################################
            //! The MemBufHost native pointer get trait specialization.
            //#############################################################################
            template<
                typename TElement, 
                typename TDim>
            struct GetNativePtr<
                host::detail::MemBufHost<TElement, TDim>>
            {
                static TElement const * getNativePtr(
                    host::detail::MemBufHost<TElement, TDim> const & memBuf)
                {
                    return memBuf.m_spMem.get();
                }
                static TElement * getNativePtr(
                    host::detail::MemBufHost<TElement, TDim> & memBuf)
                {
                    return memBuf.m_spMem.get();
                }
            };

            //#############################################################################
            //! The host accelerators memory allocation trait specialization.
            //#############################################################################
            template<
                typename TElement, 
                typename TDim>
            struct MemAlloc<
                TElement,
                TDim,
                MemorySpaceHost
            >
            {
                template<
                    typename TExtent>
                static host::detail::MemBufHost<TElement, TDim> memAlloc(
                    TExtent const & extent)
                {
                    return host::detail::MemBufHost<TElement, TDim>(extent);
                }
            };
        }
    }

    namespace traits
    {
        namespace memory
        {
            //#############################################################################
            //! The host accelerators memory copy trait specialization.
            //!
            //! Copies from host memory into host memory.
            //#############################################################################
            template<
                typename TDim>
            struct MemCopy<
                TDim, 
                MemorySpaceHost, 
                MemorySpaceHost
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

                    auto const uiExtentElementCount(alpaka::extent::getProductOfExtents(extent));
                    assert(uiExtentElementCount<=alpaka::extent::getProductOfExtents(memBufDst));
                    assert(uiExtentElementCount<=alpaka::extent::getProductOfExtents(memBufSrc));
                    auto const uiSizeBytes(uiExtentElementCount * sizeof(alpaka::memory::GetMemElemTypeT<TMemBufDst>));

                    std::memcpy(
                        reinterpret_cast<void *>(alpaka::memory::getNativePtr(memBufDst)),
                        reinterpret_cast<void const *>(alpaka::memory::getNativePtr(memBufSrc)),
                        uiSizeBytes);
                }
            };

            //#############################################################################
            //! The host accelerators memory set trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct MemSet<
                TDim, 
                MemorySpaceHost
            >
            {
                template<
                    typename TExtent, 
                    typename TMemBuf>
                static void memSet(
                    TMemBuf & memBuf, 
                    int const & iValue, 
                    TExtent const & extent)
                {
                    static_assert(
                        std::is_same<alpaka::dim::GetDimT<TMemBuf>, alpaka::dim::GetDimT<TExtent>>::value,
                        "The destination buffer and the extent are required to have the same dimensionality!");

                    auto const uiExtentElementCount(alpaka::extent::getProductOfExtents(extent));
                    assert(uiExtentElementCount<=alpaka::extent::getProductOfExtents(memBuf));
                    auto const uiSizeBytes(uiExtentElementCount * sizeof(alpaka::memory::GetMemElemTypeT<TMemBuf>));

                    std::memset(
                        reinterpret_cast<void *>(alpaka::memory::getNativePtr(memBuf)),
                        iValue,
                        uiSizeBytes);
                }
            };
        }
    }
}
