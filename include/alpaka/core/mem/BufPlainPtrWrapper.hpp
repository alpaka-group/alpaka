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
#include <alpaka/traits/Extents.hpp>    // traits::getXXX

namespace alpaka
{
    namespace mem
    {
        //#############################################################################
        //! The memory buffer wrapper used to wrap plain pointers.
        //#############################################################################
        template<
            typename TSpace,
            typename TElem,
            typename TDim>
        class BufPlainPtrWrapper
        {
        private:
            using MemSpace = SpaceT<TSpace>;
            using Elem = TElem;
            using Dim = TDim;

        public:
            //-----------------------------------------------------------------------------
            //! Constructor
            //-----------------------------------------------------------------------------
            template<
                typename TExtents>
            BufPlainPtrWrapper(
                TElem * pMem,
                TExtents const & extents) :
                    m_vExtentsElements(extents),
                    m_pMem(pMem),
                    m_uiPitchBytes(extent::getWidth(extents) * sizeof(TElem))
            {}

            //-----------------------------------------------------------------------------
            //! Constructor
            //-----------------------------------------------------------------------------
            template<
                typename TExtents>
            BufPlainPtrWrapper(
                TElem * pMem,
                UInt const & uiPitch,
                TExtents const & extents) :
                    m_vExtentsElements(extents),
                    m_pMem(pMem),
                    m_uiPitchBytes(uiPitch)
            {}

        public:
            Vec<TDim::value> m_vExtentsElements;
            TElem * m_pMem;
            UInt m_uiPitchBytes;
        };
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for BufPlainPtrWrapper.
    //-----------------------------------------------------------------------------
    namespace traits
    {
        namespace dim
        {
            //#############################################################################
            //! The BufPlainPtrWrapper dimension getter trait.
            //#############################################################################
            template<
                typename TSpace,
                typename TElem,
                typename TDim>
            struct DimType<
                alpaka::mem::BufPlainPtrWrapper<TSpace, TElem, TDim>>
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
                typename TElem,
                typename TDim>
            struct GetExtents<
                alpaka::mem::BufPlainPtrWrapper<TSpace, TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static Vec<TDim::value> getExtents(
                    alpaka::mem::BufPlainPtrWrapper<TSpace, TElem, TDim> const & extents)
                {
                    return {extents.m_vExtentsElements};
                }
            };

            //#############################################################################
            //! The BufPlainPtrWrapper width get trait specialization.
            //#############################################################################
            template<
                typename TSpace,
                typename TElem,
                typename TDim>
            struct GetWidth<
                alpaka::mem::BufPlainPtrWrapper<TSpace, TElem, TDim>,
                typename std::enable_if<(TDim::value >= 1u) && (TDim::value <= 3u)>::type>
            {
                static UInt getWidth(
                    alpaka::mem::BufPlainPtrWrapper<TSpace, TElem, TDim> const & extent)
                {
                    return extent.m_vExtentsElements[0u];
                }
            };

            //#############################################################################
            //! The BufPlainPtrWrapper height get trait specialization.
            //#############################################################################
            template<
                typename TSpace,
                typename TElem,
                typename TDim>
            struct GetHeight<
                alpaka::mem::BufPlainPtrWrapper<TSpace, TElem, TDim>,
                typename std::enable_if<(TDim::value >= 2u) && (TDim::value <= 3u)>::type>
            {
                static UInt getHeight(
                    alpaka::mem::BufPlainPtrWrapper<TSpace, TElem, TDim> const & extent)
                {
                    return extent.m_vExtentsElements[1u];
                }
            };
            //#############################################################################
            //! The BufPlainPtrWrapper depth get trait specialization.
            //#############################################################################
            template<
                typename TSpace,
                typename TElem,
                typename TDim>
            struct GetDepth<
                alpaka::mem::BufPlainPtrWrapper<TSpace, TElem, TDim>,
                typename std::enable_if<(TDim::value >= 3u) && (TDim::value <= 3u)>::type>
            {
                static UInt getDepth(
                    alpaka::mem::BufPlainPtrWrapper<TSpace, TElem, TDim> const & extent)
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
                typename TElem,
                typename TDim>
            struct GetOffsets<
                alpaka::mem::BufPlainPtrWrapper<TSpace, TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static Vec<TDim::value> getOffsets(
                    alpaka::mem::BufPlainPtrWrapper<TSpace, TElem, TDim> const &)
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
                typename TElem,
                typename TDim>
            struct SpaceType<
                alpaka::mem::BufPlainPtrWrapper<TSpace, TElem, TDim>>
            {
                using type = TSpace;
            };

            //#############################################################################
            //! The BufPlainPtrWrapper memory element type get trait specialization.
            //#############################################################################
            template<
                typename TSpace,
                typename TElem,
                typename TDim>
            struct ElemType<
                alpaka::mem::BufPlainPtrWrapper<TSpace, TElem, TDim>>
            {
                using type = TElem;
            };

            //#############################################################################
            //! The BufPlainPtrWrapper base buffer trait specialization.
            //#############################################################################
            template<
                typename TSpace,
                typename TElem,
                typename TDim>
            struct GetBuf<
                alpaka::mem::BufPlainPtrWrapper<TSpace, TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static alpaka::mem::BufPlainPtrWrapper<TSpace, TElem, TDim> const & getBuf(
                    alpaka::mem::BufPlainPtrWrapper<TSpace, TElem, TDim> const & buf)
                {
                    return buf;
                }
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static alpaka::mem::BufPlainPtrWrapper<TSpace, TElem, TDim> & getBuf(
                    alpaka::mem::BufPlainPtrWrapper<TSpace, TElem, TDim> & buf)
                {
                    return buf;
                }
            };

            //#############################################################################
            //! The BufPlainPtrWrapper native pointer get trait specialization.
            //#############################################################################
            template<
                typename TSpace,
                typename TElem,
                typename TDim>
            struct GetNativePtr<
                alpaka::mem::BufPlainPtrWrapper<TSpace, TElem, TDim>>
            {
                static TElem const * getNativePtr(
                    alpaka::mem::BufPlainPtrWrapper<TSpace, TElem, TDim> const & buf)
                {
                    return buf.m_pMem;
                }
                static TElem * getNativePtr(
                    alpaka::mem::BufPlainPtrWrapper<TSpace, TElem, TDim> & buf)
                {
                    return buf.m_pMem;
                }
            };

            //#############################################################################
            //! The BufPlainPtrWrapper memory pitch get trait specialization.
            //#############################################################################
            template<
                typename TSpace,
                typename TElem,
                typename TDim>
            struct GetPitchBytes<
                alpaka::mem::BufPlainPtrWrapper<TSpace, TElem, TDim>>
            {
                static UInt getPitchBytes(
                    alpaka::mem::BufPlainPtrWrapper<TSpace, TElem, TDim> const & pitch)
                {
                    return pitch.m_uiPitchBytes;
                }
            };
        }
    }
}