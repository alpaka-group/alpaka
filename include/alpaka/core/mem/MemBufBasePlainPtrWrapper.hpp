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

#include <alpaka/host/MemSpace.hpp>     // mem::MemSpaceHost

#include <alpaka/core/BasicDims.hpp>    // dim::Dim<N>
#include <alpaka/core/Vec.hpp>          // Vec<N>

#include <alpaka/traits/Mem.hpp>        // traits::MemCopy, ...
#include <alpaka/traits/Extents.hpp>    // traits::getXXX

namespace alpaka
{
    namespace mem
    {
        //#############################################################################
        //! The memory buffer wrapper used to wrap plain pointers.
        //#############################################################################
        template<
            typename TMemSpace,
            typename TElem,
            typename TDim>
        class MemBufBasePlainPtrWrapper
        {
        private:
            using MemSpace = TMemSpace;
            using Elem = TElem;
            using Dim = TDim;

        public:
            //-----------------------------------------------------------------------------
            //! Constructor
            //-----------------------------------------------------------------------------
            template<
                typename TExtents>
            MemBufBasePlainPtrWrapper(
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
            MemBufBasePlainPtrWrapper(
                TElem * pMem,
                std::size_t const & uiPitch,
                TExtents const & extents) :
                    m_vExtentsElements(extents),
                    m_pMem(pMem),
                    m_uiPitchBytes(uiPitch)
            {}

        public:
            Vec<TDim::value> m_vExtentsElements;
            TElem * m_pMem;
            std::size_t m_uiPitchBytes;
        };
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for MemBufBasePlainPtrWrapper.
    //-----------------------------------------------------------------------------
    namespace traits
    {
        namespace dim
        {
            //#############################################################################
            //! The MemBufBasePlainPtrWrapper dimension getter trait.
            //#############################################################################
            template<
                typename TMemSpace,
                typename TElem,
                typename TDim>
            struct DimType<
                alpaka::mem::MemBufBasePlainPtrWrapper<TMemSpace, TElem, TDim>>
            {
                using type = TDim;
            };
        }

        namespace extent
        {
            //#############################################################################
            //! The MemBufBasePlainPtrWrapper width get trait specialization.
            //#############################################################################
            template<
                typename TMemSpace,
                typename TElem,
                typename TDim>
            struct GetWidth<
                alpaka::mem::MemBufBasePlainPtrWrapper<TMemSpace, TElem, TDim>,
                typename std::enable_if<(TDim::value >= 1u) && (TDim::value <= 3u)>::type>
            {
                static std::size_t getWidth(
                    alpaka::mem::MemBufBasePlainPtrWrapper<TMemSpace, TElem, TDim> const & extent)
                {
                    return extent.m_vExtentsElements[0u];
                }
            };

            //#############################################################################
            //! The MemBufBasePlainPtrWrapper height get trait specialization.
            //#############################################################################
            template<
                typename TMemSpace,
                typename TElem,
                typename TDim>
            struct GetHeight<
                alpaka::mem::MemBufBasePlainPtrWrapper<TMemSpace, TElem, TDim>,
                typename std::enable_if<(TDim::value >= 2u) && (TDim::value <= 3u)>::type>
            {
                static std::size_t getHeight(
                    alpaka::mem::MemBufBasePlainPtrWrapper<TMemSpace, TElem, TDim> const & extent)
                {
                    return extent.m_vExtentsElements[1u];
                }
            };
            //#############################################################################
            //! The MemBufBasePlainPtrWrapper depth get trait specialization.
            //#############################################################################
            template<
                typename TMemSpace,
                typename TElem,
                typename TDim>
            struct GetDepth<
                alpaka::mem::MemBufBasePlainPtrWrapper<TMemSpace, TElem, TDim>,
                typename std::enable_if<(TDim::value >= 3u) && (TDim::value <= 3u)>::type>
            {
                static std::size_t getDepth(
                    alpaka::mem::MemBufBasePlainPtrWrapper<TMemSpace, TElem, TDim> const & extent)
                {
                    return extent.m_vExtentsElements[2u];
                }
            };
        }

        namespace mem
        {
            //#############################################################################
            //! The MemBufBasePlainPtrWrapper base memory buffer trait specialization.
            //#############################################################################
            template<
                typename TMemSpace,
                typename TElem,
                typename TDim>
            struct IsMemBufBase<
                alpaka::mem::MemBufBasePlainPtrWrapper<TMemSpace, TElem, TDim>>
            {
                static const bool value = true;
            };

            //#############################################################################
            //! The MemBufBasePlainPtrWrapper memory space trait specialization.
            //#############################################################################
            template<
                typename TMemSpace,
                typename TElem,
                typename TDim>
            struct MemSpaceType<
                alpaka::mem::MemBufBasePlainPtrWrapper<TMemSpace, TElem, TDim>>
            {
                using type = alpaka::mem::MemSpaceHost;
            };

            //#############################################################################
            //! The MemBufBasePlainPtrWrapper memory element type get trait specialization.
            //#############################################################################
            template<
                typename TMemSpace,
                typename TElem,
                typename TDim>
            struct MemElemType<
                alpaka::mem::MemBufBasePlainPtrWrapper<TMemSpace, TElem, TDim>>
            {
                using type = TElem;
            };

            //#############################################################################
            //! The MemBufBasePlainPtrWrapper base buffer trait specialization.
            //#############################################################################
            template<
                typename TMemSpace,
                typename TElem,
                typename TDim>
            struct GetMemBufBase<
                alpaka::mem::MemBufBasePlainPtrWrapper<TMemSpace, TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static alpaka::mem::MemBufBasePlainPtrWrapper<TMemSpace, TElem, TDim> const & getMemBufBase(
                    alpaka::mem::MemBufBasePlainPtrWrapper<TMemSpace, TElem, TDim> const & memBufBase)
                {
                    return memBufBase;
                }
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static alpaka::mem::MemBufBasePlainPtrWrapper<TMemSpace, TElem, TDim> & getMemBufBase(
                    alpaka::mem::MemBufBasePlainPtrWrapper<TMemSpace, TElem, TDim> & memBufBase)
                {
                    return memBufBase;
                }
            };

            //#############################################################################
            //! The MemBufBasePlainPtrWrapper native pointer get trait specialization.
            //#############################################################################
            template<
                typename TMemSpace,
                typename TElem,
                typename TDim>
            struct GetNativePtr<
                alpaka::mem::MemBufBasePlainPtrWrapper<TMemSpace, TElem, TDim>>
            {
                static TElem const * getNativePtr(
                    alpaka::mem::MemBufBasePlainPtrWrapper<TMemSpace, TElem, TDim> const & memBuf)
                {
                    return memBuf.m_pMem;
                }
                static TElem * getNativePtr(
                    alpaka::mem::MemBufBasePlainPtrWrapper<TMemSpace, TElem, TDim> & memBuf)
                {
                    return memBuf.m_pMem;
                }
            };

            //#############################################################################
            //! The MemBufBasePlainPtrWrapper memory pitch get trait specialization.
            //#############################################################################
            template<
                typename TMemSpace,
                typename TElem,
                typename TDim>
            struct GetPitchBytes<
                alpaka::mem::MemBufBasePlainPtrWrapper<TMemSpace, TElem, TDim>>
            {
                static std::size_t getPitchBytes(
                    alpaka::mem::MemBufBasePlainPtrWrapper<TMemSpace, TElem, TDim> const & memPitch)
                {
                    return memPitch.m_uiPitchBytes;
                }
            };
        }
    }
}