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

#include <alpaka/host/mem/Space.hpp>    // mem::SpaceHost

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
            typename TSpace,
            typename TDim,
            typename TElem,
            typename TDev>
        class BufPlainPtrWrapper
        {
        private:
            using MemSpace = SpaceT<TSpace>;
            using Dim = TDim;
            using Elem = TElem;
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
                    m_vExtentsElements(Vec<TDim::value>::fromExtents(extents)),
                    m_uiPitchBytes(extent::getWidth(extents) * sizeof(TElem))
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
                    m_vExtentsElements(Vec<TDim::value>::fromExtents(extents)),
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
            Vec<TDim::value> m_vExtentsElements;
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
                typename TSpace,
                typename TDim,
                typename TElem,
                typename TDev>
            struct DevType<
                alpaka::mem::BufPlainPtrWrapper<TSpace, TElem, TDim, TDev>>
            {
                using type = TDev;
            };

            //#############################################################################
            //! The BufPlainPtrWrapper device get trait specialization.
            //#############################################################################
            template<
                typename TSpace,
                typename TDim,
                typename TElem,
                typename TDev>
            struct GetDev<
                alpaka::mem::BufPlainPtrWrapper<TSpace, TDim, TElem, TDev>>
            {
                ALPAKA_FCT_HOST_ACC static auto getDev(
                    alpaka::mem::BufPlainPtrWrapper<TSpace, TDim, TElem, TDev> const & buf)
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
                typename TSpace,
                typename TDim,
                typename TElem,
                typename TDev>
            struct DimType<
                alpaka::mem::BufPlainPtrWrapper<TSpace, TDim, TElem, TDev>>
            {
                using type = TDim;
            };
        }

        namespace extent
        {
            //#############################################################################
            //! The BufPlainPtrWrapper extents get trait specialization.
            //#############################################################################
            template<
                typename TSpace,
                typename TDim,
                typename TElem,
                typename TDev>
            struct GetExtents<
                alpaka::mem::BufPlainPtrWrapper<TSpace, TDim, TElem, TDev>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST_ACC static auto getExtents(
                    alpaka::mem::BufPlainPtrWrapper<TSpace, TDim, TElem, TDev> const & extents)
                -> Vec<TDim::value>
                {
                    return {extents.m_vExtentsElements};
                }
            };

            //#############################################################################
            //! The BufPlainPtrWrapper width get trait specialization.
            //#############################################################################
            template<
                typename TSpace,
                typename TDim,
                typename TElem,
                typename TDev>
            struct GetWidth<
                alpaka::mem::BufPlainPtrWrapper<TSpace, TDim, TElem, TDev>,
                typename std::enable_if<(TDim::value >= 1u) && (TDim::value <= 3u)>::type>
            {
                ALPAKA_FCT_HOST_ACC static auto getWidth(
                    alpaka::mem::BufPlainPtrWrapper<TSpace, TDim, TElem, TDev> const & extent)
                -> UInt
                {
                    return extent.m_vExtentsElements[0u];
                }
            };

            //#############################################################################
            //! The BufPlainPtrWrapper height get trait specialization.
            //#############################################################################
            template<
                typename TSpace,
                typename TDim,
                typename TElem,
                typename TDev>
            struct GetHeight<
                alpaka::mem::BufPlainPtrWrapper<TSpace, TDim, TElem, TDev>,
                typename std::enable_if<(TDim::value >= 2u) && (TDim::value <= 3u)>::type>
            {
                ALPAKA_FCT_HOST_ACC static auto getHeight(
                    alpaka::mem::BufPlainPtrWrapper<TSpace, TDim, TElem, TDev> const & extent)
                -> UInt
                {
                    return extent.m_vExtentsElements[1u];
                }
            };
            //#############################################################################
            //! The BufPlainPtrWrapper depth get trait specialization.
            //#############################################################################
            template<
                typename TSpace,
                typename TDim,
                typename TElem,
                typename TDev>
            struct GetDepth<
                alpaka::mem::BufPlainPtrWrapper<TSpace, TDim, TElem, TDev>,
                typename std::enable_if<(TDim::value >= 3u) && (TDim::value <= 3u)>::type>
            {
                ALPAKA_FCT_HOST_ACC static auto getDepth(
                    alpaka::mem::BufPlainPtrWrapper<TSpace, TDim, TElem, TDev> const & extent)
                -> UInt
                {
                    return extent.m_vExtentsElements[2u];
                }
            };
        }

        namespace offset
        {
            //#############################################################################
            //! The BufPlainPtrWrapper offsets get trait specialization.
            //#############################################################################
            template<
                typename TSpace,
                typename TDim,
                typename TElem,
                typename TDev>
            struct GetOffsets<
                alpaka::mem::BufPlainPtrWrapper<TSpace, TDim, TElem, TDev>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST_ACC static auto getOffsets(
                    alpaka::mem::BufPlainPtrWrapper<TSpace, TDim, TElem, TDev> const &)
                -> Vec<TDim::value>
                {
                    return Vec<TDim::value>();
                }
            };
        }

        namespace mem
        {
            //#############################################################################
            //! The BufPlainPtrWrapper memory space trait specialization.
            //#############################################################################
            template<
                typename TSpace,
                typename TDim,
                typename TElem,
                typename TDev>
            struct SpaceType<
                alpaka::mem::BufPlainPtrWrapper<TSpace, TDim, TElem, TDev>>
            {
                using type = TSpace;
            };

            //#############################################################################
            //! The BufPlainPtrWrapper memory element type get trait specialization.
            //#############################################################################
            template<
                typename TSpace,
                typename TDim,
                typename TElem,
                typename TDev>
            struct ElemType<
                alpaka::mem::BufPlainPtrWrapper<TSpace, TDim, TElem, TDev>>
            {
                using type = TElem;
            };

            //#############################################################################
            //! The BufPlainPtrWrapper base buffer trait specialization.
            //#############################################################################
            template<
                typename TSpace,
                typename TDim,
                typename TElem,
                typename TDev>
            struct GetBuf<
                alpaka::mem::BufPlainPtrWrapper<TSpace, TDim, TElem, TDev>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST_ACC static auto getBuf(
                    alpaka::mem::BufPlainPtrWrapper<TSpace, TDim, TElem, TDev> const & buf)
                -> alpaka::mem::BufPlainPtrWrapper<TSpace, TDim, TElem, TDev> const &
                {
                    return buf;
                }
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST_ACC static auto getBuf(
                    alpaka::mem::BufPlainPtrWrapper<TSpace, TDim, TElem, TDev> & buf)
                -> alpaka::mem::BufPlainPtrWrapper<TSpace, TDim, TElem, TDev> &
                {
                    return buf;
                }
            };

            //#############################################################################
            //! The BufPlainPtrWrapper native pointer get trait specialization.
            //#############################################################################
            template<
                typename TSpace,
                typename TDim,
                typename TElem,
                typename TDev>
            struct GetNativePtr<
                alpaka::mem::BufPlainPtrWrapper<TSpace, TDim, TElem, TDev>>
            {
                ALPAKA_FCT_HOST_ACC static auto getNativePtr(
                    alpaka::mem::BufPlainPtrWrapper<TSpace, TDim, TElem, TDev> const & buf)
                -> TElem const *
                {
                    return buf.m_pMem;
                }
                ALPAKA_FCT_HOST_ACC static auto getNativePtr(
                    alpaka::mem::BufPlainPtrWrapper<TSpace, TDim, TElem, TDev> & buf)
                -> TElem *
                {
                    return buf.m_pMem;
                }
            };

            //#############################################################################
            //! The BufPlainPtrWrapper memory pitch get trait specialization.
            //#############################################################################
            template<
                typename TSpace,
                typename TDim,
                typename TElem,
                typename TDev>
            struct GetPitchBytes<
                alpaka::mem::BufPlainPtrWrapper<TSpace, TDim, TElem, TDev>>
            {
                ALPAKA_FCT_HOST_ACC static auto getPitchBytes(
                    alpaka::mem::BufPlainPtrWrapper<TSpace, TDim, TElem, TDev> const & pitch)
                -> UInt
                {
                    return pitch.m_uiPitchBytes;
                }
            };
        }
    }
}