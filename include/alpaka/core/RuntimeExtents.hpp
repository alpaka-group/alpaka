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

#include <alpaka/traits/Dim.hpp>        // traits::getDim
#include <alpaka/traits/Extent.hpp>     // traits::getWidth, ...

#include <alpaka/core/BasicDims.hpp>    // dim::Dim<N>

#include <cstdint>                      // std::size_t
#include <type_traits>                  // std::enable_if

namespace alpaka
{
    namespace extent
    {
        //#############################################################################
        //! The abstract runtime extent.
        //#############################################################################
        template<
            typename TDim>
        class RuntimeExtents;
        //#############################################################################
        //! The 1D runtime extent.
        //#############################################################################
        template<>
        class RuntimeExtents<
            dim::Dim1>
        {
        public:
            using Dim = dim::Dim1;

        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            template<
                typename TExtent>
            explicit RuntimeExtents(
                TExtent const & extent) :
                    m_uiWidth(extent::getWidth(extent))
            {}
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            explicit RuntimeExtents(
                std::size_t const & uiWidth) :
                    m_uiWidth(uiWidth)
            {}

        public:
            std::size_t m_uiWidth;  //!< The width of each row in elements.
        };
        //#############################################################################
        //! The 2D runtime extent.
        //#############################################################################
        template<>
        class RuntimeExtents<
            dim::Dim2>
        {
        public:
            using Dim = dim::Dim2;

        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            template<
                typename TExtent>
            explicit RuntimeExtents(
                TExtent const & extent) :
                    m_uiWidth(extent::getWidth(extent)),
                    m_uiHeight(extent::getHeight(extent))
            {}
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            explicit RuntimeExtents(
                std::size_t const & uiWidth,
                std::size_t const & uiHeight = 1) :
                    m_uiWidth(uiWidth),
                    m_uiHeight(uiHeight)
            {}

        public:
            std::size_t m_uiWidth;  //!< The width of each row in elements.
            std::size_t m_uiHeight; //!< The height of each 2D array in rows.
        };
        //#############################################################################
        //! The 3D runtime extent.
        //#############################################################################
        template<>
        class RuntimeExtents<
            dim::Dim3>
        {
        public:
            using Dim = dim::Dim3;

        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            template<
                typename TExtent>
            explicit RuntimeExtents(
                TExtent const & extent) :
                    m_uiWidth(extent::getWidth(extent)),
                    m_uiHeight(extent::getHeight(extent)),
                    m_uiDepth(extent::getDepth(extent))
            {}
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            explicit RuntimeExtents(
                std::size_t const & uiWidth,
                std::size_t const & uiHeight = 1,
                std::size_t const & uiDepth = 1) :
                    m_uiWidth(uiWidth),
                    m_uiHeight(uiHeight),
                    m_uiDepth(uiDepth)
            {}

        public:
            std::size_t m_uiWidth;  //!< The width of each row in elements.
            std::size_t m_uiHeight; //!< The height of each slice in rows.
            std::size_t m_uiDepth;  //!< The depth in slices.
        };
    }

    namespace traits
    {
        namespace dim
        {
            //#############################################################################
            //! The RuntimeExtents dimension get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetDim<
                alpaka::extent::RuntimeExtents<TDim>>
            {
                using type = TDim;
            };
        }

        namespace extent
        {
            //#############################################################################
            //! The RuntimeExtents<TDim> width get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetWidth<
                alpaka::extent::RuntimeExtents<TDim>,
                typename std::enable_if<(TDim::value >= 1u) && (TDim::value <= 3u), void>::type>
            {
                static std::size_t getWidth(
                    alpaka::extent::RuntimeExtents<TDim> const & extent)
                {
                    return extent.m_uiWidth;
                }
            };

            //#############################################################################
            //! The RuntimeExtents<TDim> height get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetHeight<
                alpaka::extent::RuntimeExtents<TDim>,
                typename std::enable_if<(TDim::value >= 2u) && (TDim::value <= 3u), void>::type>
            {
                static std::size_t getHeight(
                    alpaka::extent::RuntimeExtents<TDim> const & extent)
                {
                    return extent.m_uiHeight;
                }
            };

            //#############################################################################
            //! The RuntimeExtents<Dim3> depth get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetDepth<
                alpaka::extent::RuntimeExtents<TDim>,
                typename std::enable_if<(TDim::value >= 3u) && (TDim::value <= 3u), void>::type>
            {
                static std::size_t getDepth(
                    alpaka::extent::RuntimeExtents<TDim> const & extent)
                {
                    return extent.m_uiDepth;
                }
            };
        }
    }
}