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

#include <alpaka/host/mem/Space.hpp>        // SpaceHost
#include <alpaka/host/mem/Set.hpp>          // Set
#include <alpaka/host/Stream.hpp>           // StreamHost

#include <alpaka/core/BasicDims.hpp>        // dim::Dim<N>
#include <alpaka/core/Vec.hpp>              // Vec<TDim::value>

#include <alpaka/traits/mem/Buf.hpp>        // traits::Alloc, ...
#include <alpaka/traits/Extents.hpp>        // traits::getXXX

#include <cassert>                          // assert
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
            class BufHost
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
                ALPAKA_FCT_HOST BufHost(
                    TExtents const & extents) :
                        m_vExtentsElements(Vec<TDim::value>::fromExtents(extents)),
                        m_spMem(new TElem[computeElementCount(extents)], &BufHost::freeBuffer),
                        m_uiPitchBytes(extent::getWidth(extents) * sizeof(TElem))
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    std::cout << BOOST_CURRENT_FUNCTION
                        << " e: " << m_vExtentsElements
                        << " ptr: " << static_cast<void *>(m_spMem.get())
                        << " pitch: " << m_uiPitchBytes
                        << std::endl;
#endif
                }

            private:
                //-----------------------------------------------------------------------------
                //! \return The number of elements to allocate.
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents>
                ALPAKA_FCT_HOST static UInt computeElementCount(
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
                UInt m_uiPitchBytes;
            };
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for host::detail::BufHost.
    //-----------------------------------------------------------------------------
    namespace traits
    {
        namespace dim
        {
            //#############################################################################
            //! The BufHost dimension getter trait.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct DimType<
                host::detail::BufHost<TElem, TDim>>
            {
                using type = TDim;
            };
        }

        namespace extent
        {
            //#############################################################################
            //! The BufHost extents get trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct GetExtents<
                host::detail::BufHost<TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static Vec<TDim::value> getExtents(
                    host::detail::BufHost<TElem, TDim> const & extents)
                {
                    return {extents.m_vExtentsElements};
                }
            };

            //#############################################################################
            //! The BufHost width get trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct GetWidth<
                host::detail::BufHost<TElem, TDim>,
                typename std::enable_if<(TDim::value >= 1u) && (TDim::value <= 3u)>::type>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static UInt getWidth(
                    host::detail::BufHost<TElem, TDim> const & extent)
                {
                    return extent.m_vExtentsElements[0u];
                }
            };

            //#############################################################################
            //! The BufHost height get trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct GetHeight<
                host::detail::BufHost<TElem, TDim>,
                typename std::enable_if<(TDim::value >= 2u) && (TDim::value <= 3u)>::type>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static UInt getHeight(
                    host::detail::BufHost<TElem, TDim> const & extent)
                {
                    return extent.m_vExtentsElements[1u];
                }
            };
            //#############################################################################
            //! The BufHost depth get trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct GetDepth<
                host::detail::BufHost<TElem, TDim>,
                typename std::enable_if<(TDim::value >= 3u) && (TDim::value <= 3u)>::type>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static UInt getDepth(
                    host::detail::BufHost<TElem, TDim> const & extent)
                {
                    return extent.m_vExtentsElements[2u];
                }
            };
        }
        
        namespace offset
        {
            //#############################################################################
            //! The BufHost offsets get trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct GetOffsets<
                host::detail::BufHost<TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static Vec<TDim::value> getOffsets(
                    host::detail::BufHost<TElem, TDim> const &)
                {
                    return Vec<TDim::value>();
                }
            };
        }

        namespace mem
        {
            //#############################################################################
            //! The BufHost memory space trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct SpaceType<
                host::detail::BufHost<TElem, TDim>>
            {
                using type = alpaka::mem::SpaceHost;
            };

            //#############################################################################
            //! The BufHost memory element type get trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct ElemType<
                host::detail::BufHost<TElem, TDim>>
            {
                using type = TElem;
            };

            //#############################################################################
            //! The BufHost base buffer trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct GetBuf<
                host::detail::BufHost<TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static host::detail::BufHost<TElem, TDim> const & getBuf(
                    host::detail::BufHost<TElem, TDim> const & buf)
                {
                    return buf;
                }
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static host::detail::BufHost<TElem, TDim> & getBuf(
                    host::detail::BufHost<TElem, TDim> & buf)
                {
                    return buf;
                }
            };

            //#############################################################################
            //! The BufHost native pointer get trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct GetNativePtr<
                host::detail::BufHost<TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static TElem const * getNativePtr(
                    host::detail::BufHost<TElem, TDim> const & buf)
                {
                    return buf.m_spMem.get();
                }
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static TElem * getNativePtr(
                    host::detail::BufHost<TElem, TDim> & buf)
                {
                    return buf.m_spMem.get();
                }
            };

            //#############################################################################
            //! The BufHost pitch get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct GetPitchBytes<
                host::detail::BufHost<TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static UInt getPitchBytes(
                    host::detail::BufHost<TElem, TDim> const & pitch)
                {
                    // No pitch on the host currently.
                    return pitch.m_uiPitchBytes;
                }
            };

            //#############################################################################
            //! The host accelerators memory allocation trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct Alloc<
                TElem,
                TDim,
                alpaka::mem::SpaceHost>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents>
                ALPAKA_FCT_HOST static host::detail::BufHost<TElem, TDim> alloc(
                    TExtents const & extents)
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    return host::detail::BufHost<TElem, TDim>(extents);
                }
            };
        }
    }
}
