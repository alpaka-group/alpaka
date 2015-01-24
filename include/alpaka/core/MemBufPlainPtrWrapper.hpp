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

#include <alpaka/traits/Mem.hpp>            // traits::MemCopy, ...
#include <alpaka/traits/Extents.hpp>        // traits::getXXX

#include <alpaka/core/BasicDims.hpp>        // dim::Dim<N>
#include <alpaka/core/BasicExtents.hpp>     // extent::BasicExtents<TDim>

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
        class MemBufPlainPtrWrapper :
            public extent::BasicExtents<TDim>
        {
        public:
            using MemSpace = TMemSpace;
            using Elem = TElem;
            using Dim = TDim;

        public:
            //-----------------------------------------------------------------------------
            //! Constructor
            //-----------------------------------------------------------------------------
            template<
                typename TExtents>
                MemBufPlainPtrWrapper(
                TElem * pMem,
                TExtents const & extents) :
                extent::BasicExtents<TDim>(extents),
                    m_pMem(pMem),
                    m_uiPitchBytes(extent::getWidth(extents) * sizeof(TElem))
            {}

            //-----------------------------------------------------------------------------
            //! Constructor
            //-----------------------------------------------------------------------------
            template<
                typename TExtents>
                MemBufPlainPtrWrapper(
                TElem * pMem,
                std::size_t const & uiPitch,
                TExtents const & extents) :
                    extent::BasicExtents<TDim>(extent),
                    m_pMem(pMem),
                    m_uiPitchBytes(uiPitch)
            {}

        public:
            TElem * m_pMem;
            std::size_t m_uiPitchBytes;
        };
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for MemBufPlainPtrWrapper.
    //-----------------------------------------------------------------------------
    namespace traits
    {
        namespace dim
        {
            //#############################################################################
            //! The MemBufPlainPtrWrapper dimension getter trait.
            //#############################################################################
            template<
                typename TMemSpace,
                typename TElem,
                typename TDim>
            struct GetDim<
                alpaka::mem::MemBufPlainPtrWrapper<TMemSpace, TElem, TDim>>
            {
                using type = TDim;
            };
        }

        namespace extent
        {
            //#############################################################################
            //! The MemBufPlainPtrWrapper width get trait specialization.
            //#############################################################################
            template<
                typename TMemSpace,
                typename TElem,
                typename TDim>
            struct GetWidth<
                alpaka::mem::MemBufPlainPtrWrapper<TMemSpace, TElem, TDim>,
                typename std::enable_if<(TDim::value >= 1u) && (TDim::value <= 3u)>::type>
            {
                static std::size_t getWidth(
                    alpaka::mem::MemBufPlainPtrWrapper<TMemSpace, TElem, TDim> const & extent)
                {
                    return extent.m_uiWidth;
                }
            };

            //#############################################################################
            //! The MemBufPlainPtrWrapper height get trait specialization.
            //#############################################################################
            template<
                typename TMemSpace,
                typename TElem,
                typename TDim>
            struct GetHeight<
                alpaka::mem::MemBufPlainPtrWrapper<TMemSpace, TElem, TDim>,
                typename std::enable_if<(TDim::value >= 2u) && (TDim::value <= 3u)>::type>
            {
                static std::size_t getHeight(
                    alpaka::mem::MemBufPlainPtrWrapper<TMemSpace, TElem, TDim> const & extent)
                {
                    return extent.m_uiHeight;
                }
            };
            //#############################################################################
            //! The MemBufPlainPtrWrapper depth get trait specialization.
            //#############################################################################
            template<
                typename TMemSpace,
                typename TElem,
                typename TDim>
            struct GetDepth<
                alpaka::mem::MemBufPlainPtrWrapper<TMemSpace, TElem, TDim>,
                typename std::enable_if<(TDim::value >= 3u) && (TDim::value <= 3u)>::type>
            {
                static std::size_t getDepth(
                    alpaka::mem::MemBufPlainPtrWrapper<TMemSpace, TElem, TDim> const & extent)
                {
                    return extent.m_uiDepth;
                }
            };
        }

        namespace mem
        {
            //#############################################################################
            //! The MemBufPlainPtrWrapper memory space trait specialization.
            //#############################################################################
            template<
                typename TMemSpace,
                typename TElem,
                typename TDim>
            struct GetMemSpace<
                alpaka::mem::MemBufPlainPtrWrapper<TMemSpace, TElem, TDim>>
            {
                using type = alpaka::mem::MemSpaceHost;
            };

            //#############################################################################
            //! The MemBufPlainPtrWrapper memory element type get trait specialization.
            //#############################################################################
            template<
                typename TMemSpace,
                typename TElem,
                typename TDim>
            struct GetMemElem<
                alpaka::mem::MemBufPlainPtrWrapper<TMemSpace, TElem, TDim>>
            {
                using type = TElem;
            };

            //#############################################################################
            //! The MemBufPlainPtrWrapper native pointer get trait specialization.
            //#############################################################################
            template<
                typename TMemSpace,
                typename TElem,
                typename TDim>
            struct GetNativePtr<
                alpaka::mem::MemBufPlainPtrWrapper<TMemSpace, TElem, TDim>>
            {
                static TElem const * getNativePtr(
                    alpaka::mem::MemBufPlainPtrWrapper<TMemSpace, TElem, TDim> const & memBuf)
                {
                    return memBuf.m_pMem;
                }
                static TElem * getNativePtr(
                    alpaka::mem::MemBufPlainPtrWrapper<TMemSpace, TElem, TDim> & memBuf)
                {
                    return memBuf.m_pMem;
                }
            };

            //#############################################################################
            //! The MemBufPlainPtrWrapper memory pitch get trait specialization.
            //#############################################################################
            template<
                typename TMemSpace,
                typename TElem,
                typename TDim>
            struct GetPitchBytes<
                alpaka::mem::MemBufPlainPtrWrapper<TMemSpace, TElem, TDim>>
            {
                static std::size_t getPitchBytes(
                    alpaka::mem::MemBufPlainPtrWrapper<TMemSpace, TElem, TDim> const & memPitch)
                {
                    return memPitch.m_uiPitchBytes;
                }
            };
        }
    }
}