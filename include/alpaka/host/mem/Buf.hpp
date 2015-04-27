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

#include <alpaka/host/mem/Space.hpp>    // SpaceHost
#include <alpaka/host/mem/Set.hpp>      // Set
#include <alpaka/host/Dev.hpp>

#include <alpaka/core/mem/View.hpp>     // View
#include <alpaka/core/BasicDims.hpp>    // dim::Dim<N>
#include <alpaka/core/Vec.hpp>          // Vec<TDim>

#include <alpaka/traits/mem/Buf.hpp>    // traits::Alloc, ...
#include <alpaka/traits/Extent.hpp>     // traits::getXXX

#include <cassert>                      // assert
#include <memory>                       // std::shared_ptr

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
                typename TDim,
                typename TDev>
            class BufHost
            {
            private:
                using Elem = TElem;
                using Dim = TDim;
                using Dev = TDev;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents>
                ALPAKA_FCT_HOST BufHost(
                    TDev const & dev,
                    TExtents const & extents) :
                        m_Dev(dev),
                        m_vExtentsElements(extent::getExtentsNd<TDim, UInt>(extents)),
                        m_spMem(new TElem[computeElementCount(extents)], &BufHost::freeBuffer),
                        m_uiPitchBytes(extent::getWidth<UInt>(extents) * sizeof(TElem))
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
                ALPAKA_FCT_HOST static auto computeElementCount(
                    TExtents const & extents)
                -> UInt
                {
                    auto const uiExtentsElementCount(extent::getProductOfExtents<UInt>(extents));
                    assert(uiExtentsElementCount>0);

                    return uiExtentsElementCount;
                }
                //-----------------------------------------------------------------------------
                //! Frees the shared buffer.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto freeBuffer(
                    TElem * pBuffer)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    assert(pBuffer);
                    delete[] pBuffer;
                }

            public:
                TDev m_Dev;
                Vec<TDim> m_vExtentsElements;
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
        namespace dev
        {
            //#############################################################################
            //! The BufCuda device type trait specialization.
            //#############################################################################
            template<
                typename TDev,
                typename TElem,
                typename TDim>
            struct DevType<
                host::detail::BufHost<TElem, TDim, TDev>>
            {
                using type = TDev;
            };

            //#############################################################################
            //! The BufHost device get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev>
            struct GetDev<
                host::detail::BufHost<TElem, TDim, TDev>>
            {
                ALPAKA_FCT_HOST static auto getDev(
                    host::detail::BufHost<TElem, TDim, TDev> const & buf)
                -> TDev
                {
                    return buf.m_Dev;
                }
            };
        }

        namespace dim
        {
            //#############################################################################
            //! The BufHost dimension getter trait.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev>
            struct DimType<
                host::detail::BufHost<TElem, TDim, TDev>>
            {
                using type = TDim;
            };
        }

        namespace extent
        {
            //#############################################################################
            //! The BufHost width get trait specialization.
            //#############################################################################
            template<
                UInt TuiIdx,
                typename TElem,
                typename TDim,
                typename TDev>
            struct GetExtent<
                TuiIdx,
                host::detail::BufHost<TElem, TDim, TDev>,
                typename std::enable_if<TDim::value >= (TuiIdx+1)>::type>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getExtent(
                    host::detail::BufHost<TElem, TDim, TDev> const & extents)
                -> UInt
                {
                    return extents.m_vExtentsElements[TuiIdx];
                }
            };
        }

        namespace offset
        {
            //#############################################################################
            //! The BufHost offset get trait specialization.
            //#############################################################################
            template<
                UInt TuiIdx,
                typename TElem,
                typename TDim,
                typename TDev>
            struct GetOffset<
                TuiIdx,
                host::detail::BufHost<TElem, TDim, TDev>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getOffset(
                    host::detail::BufHost<TElem, TDim, TDev> const &)
                -> UInt
                {
                    return 0u;
                }
            };
        }

        namespace mem
        {
            //#############################################################################
            //! The host accelerators memory buffer type trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev>
            struct BufType<
                TElem,
                TDim,
                TDev,
                typename std::enable_if<std::is_same<alpaka::mem::SpaceT<alpaka::acc::AccT<TDev>>, alpaka::mem::SpaceHost>::value>::type>
            {
                using type = host::detail::BufHost<TElem, TDim, TDev>;
            };

            //#############################################################################
            //! The host accelerators memory view type trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev>
            struct ViewType<
                TElem,
                TDim,
                TDev,
                typename std::enable_if<std::is_same<alpaka::mem::SpaceT<alpaka::acc::AccT<TDev>>, alpaka::mem::SpaceHost>::value>::type>
            {
                using type = alpaka::mem::detail::View<TElem, TDim, TDev>;
            };

            //#############################################################################
            //! The BufHost memory space trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev>
            struct SpaceType<
                host::detail::BufHost<TElem, TDim, TDev>>
            {
                using type = alpaka::mem::SpaceHost;
            };

            //#############################################################################
            //! The BufHost memory element type get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev>
            struct ElemType<
                host::detail::BufHost<TElem, TDim, TDev>>
            {
                using type = TElem;
            };

            //#############################################################################
            //! The BufHost base trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev>
            struct GetBase<
                host::detail::BufHost<TElem, TDim, TDev>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getBase(
                    host::detail::BufHost<TElem, TDim, TDev> const & buf)
                -> host::detail::BufHost<TElem, TDim, TDev> const &
                {
                    return buf;
                }
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getBase(
                    host::detail::BufHost<TElem, TDim, TDev> & buf)
                -> host::detail::BufHost<TElem, TDim, TDev> &
                {
                    return buf;
                }
            };

            //#############################################################################
            //! The BufHost native pointer get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev>
            struct GetPtrNative<
                host::detail::BufHost<TElem, TDim, TDev>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getPtrNative(
                    host::detail::BufHost<TElem, TDim, TDev> const & buf)
                -> TElem const *
                {
                    return buf.m_spMem.get();
                }
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getPtrNative(
                    host::detail::BufHost<TElem, TDim, TDev> & buf)
                -> TElem *
                {
                    return buf.m_spMem.get();
                }
            };
            //#############################################################################
            //! The BufHost pointer on device get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev>
            struct GetPtrDev<
                host::detail::BufHost<TElem, TDim, TDev>,
                TDev>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getPtrDev(
                    host::detail::BufHost<TElem, TDim, TDev> const & buf,
                    TDev const & dev)
                -> TElem const *
                {
                    if(dev == alpaka::dev::getDev(buf))
                    {
                        return buf.m_spMem.get();
                    }
                    else
                    {
                        throw std::runtime_error("The buffer is not accessible from the given device!");
                    }
                }
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getPtrDev(
                    host::detail::BufHost<TElem, TDim, TDev> & buf,
                    TDev const & dev)
                -> TElem *
                {
                    if(dev == alpaka::dev::getDev(buf))
                    {
                        return buf.m_spMem.get();
                    }
                    else
                    {
                        throw std::runtime_error("The buffer is not accessible from the given device!");
                    }
                }
            };

            //#############################################################################
            //! The BufHost pitch get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev>
            struct GetPitchBytes<
                0u,
                host::detail::BufHost<TElem, TDim, TDev>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getPitchBytes(
                    host::detail::BufHost<TElem, TDim, TDev> const & pitch)
                -> UInt
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
                typename TDim,
                typename TDev>
            struct Alloc<
                TElem,
                TDim,
                TDev,
                typename std::enable_if<std::is_same<alpaka::mem::SpaceT<alpaka::acc::AccT<TDev>>, alpaka::mem::SpaceHost>::value>::type>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents>
                ALPAKA_FCT_HOST static auto alloc(
                    TDev const & dev,
                    TExtents const & extents)
                -> host::detail::BufHost<TElem, TDim, TDev>
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    return host::detail::BufHost<
                        TElem,
                        TDim,
                        TDev>(
                            dev,
                            extents);
                }
            };

            //#############################################################################
            //! The host accelerators memory mapping trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev>
            struct Map<
                host::detail::BufHost<TElem, TDim, TDev>,
                TDev>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto map(
                    host::detail::BufHost<TElem, TDim, TDev> const & buf,
                    TDev const & dev)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    if(alpaka::dev::getDev(buf) != dev)
                    {
                        throw std::runtime_error("Memory mapping of BufHost between two devices is not implemented!");
                    }
                    // If it is the same device, nothing has to be mapped.
                }
            };

            //#############################################################################
            //! The host accelerators memory unmapping trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev>
            struct Unmap<
                host::detail::BufHost<TElem, TDim, TDev>,
                TDev>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto unmap(
                    host::detail::BufHost<TElem, TDim, TDev> const & buf,
                    TDev const & dev)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    if(alpaka::dev::getDev(buf) != dev)
                    {
                        throw std::runtime_error("Memory unmapping of BufHost between two devices is not implemented!");
                    }
                    // If it is the same device, nothing has to be mapped.
                }
            };
        }
    }
}
