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

#include <alpaka/mem/buf/Traits.hpp>    // DevType, DimType, GetExtent,Copy, GetOffset, ...

#include <alpaka/core/Vec.hpp>          // Vec<N>

namespace alpaka
{
    namespace mem
    {
        namespace buf
        {
            //#############################################################################
            //! The memory buffer wrapper used to wrap plain pointers.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev>
            class BufPlainPtrWrapper final
            {
            public:
                using Elem = TElem;
                using Dim = TDim;
                using Dev = TDev;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents>
                ALPAKA_FCT_HOST_ACC BufPlainPtrWrapper(
                    TElem * pMem,
                    TDev const & dev,
                    TExtents const & extents = TExtents()) :
                        m_pMem(pMem),
                        m_Dev(dev),
                        m_vExtentsElements(extent::getExtentsVecEnd<TDim, Uint>(extents)),
                        m_uiPitchBytes(extent::getWidth<Uint>(extents) * sizeof(TElem))
                {}

                //-----------------------------------------------------------------------------
                //! Constructor
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents>
                ALPAKA_FCT_HOST_ACC BufPlainPtrWrapper(
                    TElem * pMem,
                    TDev const dev,
                    TExtents const & extents,
                    Uint const & uiPitch) :
                        m_pMem(pMem),
                        m_Dev(dev),
                        m_vExtentsElements(extent::getExtentsVecEnd<TDim, Uint>(extents)),
                        m_uiPitchBytes(uiPitch)
                {}

                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST_ACC BufPlainPtrWrapper(BufPlainPtrWrapper const &) = delete;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST_ACC BufPlainPtrWrapper(BufPlainPtrWrapper &&) = default;
                //-----------------------------------------------------------------------------
                //! Copy assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST_ACC auto operator=(BufPlainPtrWrapper const &) -> BufPlainPtrWrapper & = delete;
                //-----------------------------------------------------------------------------
                //! Move assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST_ACC auto operator=(BufPlainPtrWrapper &&) -> BufPlainPtrWrapper & = default;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST_ACC ~BufPlainPtrWrapper() = default;

            public:
                TElem * m_pMem;
                TDev m_Dev;
                Vec<TDim> m_vExtentsElements;
                Uint m_uiPitchBytes;
            };
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for BufPlainPtrWrapper.
    //-----------------------------------------------------------------------------
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The BufPlainPtrWrapper device type trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev>
            struct DevType<
                mem::buf::BufPlainPtrWrapper<TElem, TDim, TDev>>
            {
                using type = TDev;
            };

            //#############################################################################
            //! The BufPlainPtrWrapper device get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev>
            struct GetDev<
                mem::buf::BufPlainPtrWrapper<TElem, TDim, TDev>>
            {
                ALPAKA_FCT_HOST_ACC static auto getDev(
                    mem::buf::BufPlainPtrWrapper<TElem, TDim, TDev> const & buf)
                    -> TDev
                {
                    return buf.m_Dev;
                }
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The BufPlainPtrWrapper dimension getter trait.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev>
            struct DimType<
                mem::buf::BufPlainPtrWrapper<TElem, TDim, TDev>>
            {
                using type = TDim;
            };
        }
    }
    namespace extent
    {
        namespace traits
        {
            //#############################################################################
            //! The BufPlainPtrWrapper width get trait specialization.
            //#############################################################################
            template<
                typename TIdx,
                typename TElem,
                typename TDim,
                typename TDev>
            struct GetExtent<
                TIdx,
                mem::buf::BufPlainPtrWrapper<TElem, TDim, TDev>,
                typename std::enable_if<(TDim::value > TIdx::value)>::type>
            {
                ALPAKA_FCT_HOST_ACC static auto getExtent(
                    mem::buf::BufPlainPtrWrapper<TElem, TDim, TDev> const & extents)
                -> Uint
                {
                    return extents.m_vExtentsElements[TIdx::value];
                }
            };
        }
    }
    namespace mem
    {
        namespace view
        {
            namespace traits
            {
                //#############################################################################
                //! The BufPlainPtrWrapper memory element type get trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim,
                    typename TDev>
                struct ElemType<
                    buf::BufPlainPtrWrapper<TElem, TDim, TDev>>
                {
                    using type = TElem;
                };

                //#############################################################################
                //! The BufPlainPtrWrapper buf trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim,
                    typename TDev>
                struct GetBuf<
                    buf::BufPlainPtrWrapper<TElem, TDim, TDev>>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST_ACC static auto getBuf(
                        buf::BufPlainPtrWrapper<TElem, TDim, TDev> const & buf)
                    -> buf::BufPlainPtrWrapper<TElem, TDim, TDev> const &
                    {
                        return buf;
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST_ACC static auto getBuf(
                        buf::BufPlainPtrWrapper<TElem, TDim, TDev> & buf)
                    -> buf::BufPlainPtrWrapper<TElem, TDim, TDev> &
                    {
                        return buf;
                    }
                };

                //#############################################################################
                //! The BufPlainPtrWrapper native pointer get trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim,
                    typename TDev>
                struct GetPtrNative<
                    buf::BufPlainPtrWrapper<TElem, TDim, TDev>>
                {
                    ALPAKA_FCT_HOST_ACC static auto getPtrNative(
                        buf::BufPlainPtrWrapper<TElem, TDim, TDev> const & buf)
                    -> TElem const *
                    {
                        return buf.m_pMem;
                    }
                    ALPAKA_FCT_HOST_ACC static auto getPtrNative(
                        buf::BufPlainPtrWrapper<TElem, TDim, TDev> & buf)
                    -> TElem *
                    {
                        return buf.m_pMem;
                    }
                };

                //#############################################################################
                //! The BufPlainPtrWrapper memory pitch get trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim,
                    typename TDev>
                struct GetPitchBytes<
                    std::integral_constant<Uint, 0u>,
                    buf::BufPlainPtrWrapper<TElem, TDim, TDev>>
                {
                    ALPAKA_FCT_HOST_ACC static auto getPitchBytes(
                        buf::BufPlainPtrWrapper<TElem, TDim, TDev> const & buf)
                    -> Uint
                    {
                        return buf.m_uiPitchBytes;
                    }
                };
            }
        }
    }
    namespace offset
    {
        namespace traits
        {
            //#############################################################################
            //! The BufPlainPtrWrapper offset get trait specialization.
            //#############################################################################
            template<
                typename TIdx,
                typename TElem,
                typename TDim,
                typename TDev>
            struct GetOffset<
                TIdx,
                mem::buf::BufPlainPtrWrapper<TElem, TDim, TDev>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST_ACC static auto getOffset(
                    mem::buf::BufPlainPtrWrapper<TElem, TDim, TDev> const &)
                -> Uint
                {
                    return 0u;
                }
            };
        }
    }
}