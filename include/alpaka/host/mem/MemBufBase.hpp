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
#include <alpaka/host/mem/MemSet.hpp>       // MemSet
#include <alpaka/host/Stream.hpp>           // StreamHost

#include <alpaka/core/BasicDims.hpp>        // dim::Dim<N>
#include <alpaka/core/Vec.hpp>              // Vec<TDim::value>

#include <alpaka/traits/mem/MemBufBase.hpp> // traits::MemAlloc, ...
#include <alpaka/traits/Extents.hpp>        // traits::getXXX

#include <cassert>                          // assert
#include <cstddef>                          // std::size_t
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
            class MemBufBaseHost
            {
            private:
                using Elem = TElem;
                using Dim = TDim;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents>
                ALPAKA_FCT_HOST MemBufBaseHost(
                    TExtents const & extents) :
                        m_vExtentsElements(Vec<TDim::value>::fromExtents(extents)),
                        m_spMem(new TElem[computeElementCount(extents)], &MemBufBaseHost::freeBuffer),
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
                ALPAKA_FCT_HOST static std::size_t computeElementCount(
                    TExtents const & extents)
                {
                    auto const uiExtentsElementCount(extent::getProductOfExtents(extents));
                    assert(uiExtentsElementCount>0);

                    return uiExtentsElementCount;
                }
                //-----------------------------------------------------------------------------
                //! Frees the shared buffer.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static void freeBuffer(
                    TElem * pBuffer)
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    assert(pBuffer);
                    delete[] pBuffer;
                }

            public:
                Vec<TDim::value> m_vExtentsElements;
                std::shared_ptr<TElem> m_spMem;
                std::size_t m_uiPitchBytes;
            };
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for host::detail::MemBufBaseHost.
    //-----------------------------------------------------------------------------
    namespace traits
    {
        namespace dim
        {
            //#############################################################################
            //! The MemBufBaseHost dimension getter trait.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct DimType<
                host::detail::MemBufBaseHost<TElem, TDim>>
            {
                using type = TDim;
            };
        }

        namespace extent
        {
            //#############################################################################
            //! The MemBufBaseHost width get trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct GetWidth<
                host::detail::MemBufBaseHost<TElem, TDim>,
                typename std::enable_if<(TDim::value >= 1u) && (TDim::value <= 3u)>::type>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static std::size_t getWidth(
                    host::detail::MemBufBaseHost<TElem, TDim> const & extent)
                {
                    return extent.m_vExtentsElements[0u];
                }
            };

            //#############################################################################
            //! The MemBufBaseHost height get trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct GetHeight<
                host::detail::MemBufBaseHost<TElem, TDim>,
                typename std::enable_if<(TDim::value >= 2u) && (TDim::value <= 3u)>::type>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static std::size_t getHeight(
                    host::detail::MemBufBaseHost<TElem, TDim> const & extent)
                {
                    return extent.m_vExtentsElements[1u];
                }
            };
            //#############################################################################
            //! The MemBufBaseHost depth get trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct GetDepth<
                host::detail::MemBufBaseHost<TElem, TDim>,
                typename std::enable_if<(TDim::value >= 3u) && (TDim::value <= 3u)>::type>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static std::size_t getDepth(
                    host::detail::MemBufBaseHost<TElem, TDim> const & extent)
                {
                    return extent.m_vExtentsElements[2u];
                }
            };
        }

        namespace mem
        {
            //#############################################################################
            //! The MemBufBaseHost memory space trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct MemSpaceType<
                host::detail::MemBufBaseHost<TElem, TDim>>
            {
                using type = alpaka::mem::MemSpaceHost;
            };

            //#############################################################################
            //! The MemBufBaseHost memory element type get trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct MemElemType<
                host::detail::MemBufBaseHost<TElem, TDim>>
            {
                using type = TElem;
            };

            //#############################################################################
            //! The MemBufBaseHost base buffer trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct GetMemBufBase<
                host::detail::MemBufBaseHost<TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static host::detail::MemBufBaseHost<TElem, TDim> getMemBufBase(
                    host::detail::MemBufBaseHost<TElem, TDim> const & memBufBase)
                {
                    return memBufBase;
                }
            };

            //#############################################################################
            //! The MemBufBaseHost native pointer get trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct GetNativePtr<
                host::detail::MemBufBaseHost<TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static TElem const * getNativePtr(
                    host::detail::MemBufBaseHost<TElem, TDim> const & memBuf)
                {
                    return memBuf.m_spMem.get();
                }
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static TElem * getNativePtr(
                    host::detail::MemBufBaseHost<TElem, TDim> & memBuf)
                {
                    return memBuf.m_spMem.get();
                }
            };

            //#############################################################################
            //! The MemBufBaseHost pitch get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct GetPitchBytes<
                host::detail::MemBufBaseHost<TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static std::size_t getPitchBytes(
                    host::detail::MemBufBaseHost<TElem, TDim> const & memPitch)
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
                ALPAKA_FCT_HOST static host::detail::MemBufBaseHost<TElem, TDim> memAlloc(
                    TExtents const & extents)
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    return host::detail::MemBufBaseHost<TElem, TDim>(extents);
                }
            };
        }
    }
}
