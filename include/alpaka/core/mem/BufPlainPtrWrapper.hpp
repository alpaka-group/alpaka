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
#include <alpaka/core/Vec.hpp>          // Vec<N>

#include <alpaka/traits/Mem.hpp>        // traits::Copy, ...
#include <alpaka/traits/Extent.hpp>     // traits::getXXX

namespace alpaka
{
    namespace mem
    {
        //#############################################################################
        //! The memory buffer wrapper used to wrap plain pointers.
        //#############################################################################
        template<
            typename TElem,
            typename TDim,
            typename TDev>
        class BufPlainPtrWrapper
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
                    m_vExtentsElements(extent::getExtentsVecNd<TDim, UInt>(extents)),
                    m_uiPitchBytes(extent::getWidth<UInt>(extents) * sizeof(TElem))
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
                UInt const & uiPitch) :
                    m_pMem(pMem),
                    m_Dev(dev),
                    m_vExtentsElements(extent::getExtentsVecNd<TDim, UInt>(extents)),
                    m_uiPitchBytes(uiPitch)
            {}

            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST_ACC BufPlainPtrWrapper(BufPlainPtrWrapper const &) = delete;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST_ACC BufPlainPtrWrapper(BufPlainPtrWrapper &&) = default;
#endif
            //-----------------------------------------------------------------------------
            //! Copy assignment.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST_ACC auto operator=(BufPlainPtrWrapper const &) -> BufPlainPtrWrapper & = delete;
            //-----------------------------------------------------------------------------
            //! Move assignment.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST_ACC auto operator=(BufPlainPtrWrapper &&) -> BufPlainPtrWrapper & = default;
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
#if BOOST_COMP_INTEL
            ALPAKA_FCT_HOST_ACC virtual ~BufPlainPtrWrapper() = default;
#else
            ALPAKA_FCT_HOST_ACC virtual ~BufPlainPtrWrapper() noexcept = default;
#endif

        public:
            TElem * m_pMem;
            TDev m_Dev;
            Vec<TDim> m_vExtentsElements;
            UInt m_uiPitchBytes;
        };
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for BufPlainPtrWrapper.
    //-----------------------------------------------------------------------------
    namespace traits
    {
        namespace dev
        {
            //#############################################################################
            //! The BufPlainPtrWrapper device type trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev>
            struct DevType<
                alpaka::mem::BufPlainPtrWrapper<TElem, TDim, TDev>>
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
                alpaka::mem::BufPlainPtrWrapper<TElem, TDim, TDev>>
            {
                ALPAKA_FCT_HOST_ACC static auto getDev(
                    alpaka::mem::BufPlainPtrWrapper<TElem, TDim, TDev> const & buf)
                    -> TDev
                {
                    return buf.m_Dev;
                }
            };
        }

        namespace dim
        {
            //#############################################################################
            //! The BufPlainPtrWrapper dimension getter trait.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev>
            struct DimType<
                alpaka::mem::BufPlainPtrWrapper<TElem, TDim, TDev>>
            {
                using type = TDim;
            };
        }

        namespace extent
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
                alpaka::mem::BufPlainPtrWrapper<TElem, TDim, TDev>,
                typename std::enable_if<(TDim::value > TIdx::value)>::type>
            {
                ALPAKA_FCT_HOST_ACC static auto getExtent(
                    alpaka::mem::BufPlainPtrWrapper<TElem, TDim, TDev> const & extents)
                -> UInt
                {
                    return extents.m_vExtentsElements[TIdx::value];
                }
            };
        }

        namespace mem
        {
            //#############################################################################
            //! The BufPlainPtrWrapper memory element type get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev>
            struct ElemType<
                alpaka::mem::BufPlainPtrWrapper<TElem, TDim, TDev>>
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
                alpaka::mem::BufPlainPtrWrapper<TElem, TDim, TDev>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST_ACC static auto getBuf(
                    alpaka::mem::BufPlainPtrWrapper<TElem, TDim, TDev> const & buf)
                -> alpaka::mem::BufPlainPtrWrapper<TElem, TDim, TDev> const &
                {
                    return buf;
                }
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST_ACC static auto getBuf(
                    alpaka::mem::BufPlainPtrWrapper<TElem, TDim, TDev> & buf)
                -> alpaka::mem::BufPlainPtrWrapper<TElem, TDim, TDev> &
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
                alpaka::mem::BufPlainPtrWrapper<TElem, TDim, TDev>>
            {
                ALPAKA_FCT_HOST_ACC static auto getPtrNative(
                    alpaka::mem::BufPlainPtrWrapper<TElem, TDim, TDev> const & buf)
                -> TElem const *
                {
                    return buf.m_pMem;
                }
                ALPAKA_FCT_HOST_ACC static auto getPtrNative(
                    alpaka::mem::BufPlainPtrWrapper<TElem, TDim, TDev> & buf)
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
                std::integral_constant<UInt, 0u>,
                alpaka::mem::BufPlainPtrWrapper<TElem, TDim, TDev>>
            {
                ALPAKA_FCT_HOST_ACC static auto getPitchBytes(
                    alpaka::mem::BufPlainPtrWrapper<TElem, TDim, TDev> const & buf)
                -> UInt
                {
                    return buf.m_uiPitchBytes;
                }
            };
        }

        namespace offset
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
                alpaka::mem::BufPlainPtrWrapper<TElem, TDim, TDev>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST_ACC static auto getOffset(
                    alpaka::mem::BufPlainPtrWrapper<TElem, TDim, TDev> const &)
                -> UInt
                {
                    return 0u;
                }
            };
        }
    }
}