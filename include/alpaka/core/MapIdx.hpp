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

#include <alpaka/core/Vec.hpp>          // Vec
#include <alpaka/core/Common.hpp>       // ALPAKA_FN_HOST_ACC

#if !defined(__CUDA_ARCH__)
    #include <boost/core/ignore_unused.hpp> // boost::ignore_unused
#endif

namespace alpaka
{
    namespace detail
    {
        //#############################################################################
        //! Maps a linear index to a N dimensional index.
        //#############################################################################
        template<
            std::size_t TuiIdxDimOut,
            std::size_t TuiIdxDimIn>
        struct MapIdx;
        //#############################################################################
        //! Maps a linear index to a linear index.
        //#############################################################################
        template<>
        struct MapIdx<
            1u,
            1u>
        {
            //-----------------------------------------------------------------------------
            // \tparam TElem Type of the index values.
            // \param Index Idx to be mapped.
            // \param Extents Spatial size to map the index to.
            // \return Vector of dimension TuiDimDst.
            //-----------------------------------------------------------------------------
            template<
                typename TElem>
            ALPAKA_FN_HOST_ACC static auto mapIdx(
                Vec<dim::Dim<1u>, TElem> const & idx,
                Vec<dim::Dim<1u>, TElem> const & extents)
            -> Vec<dim::Dim<1u>, TElem>
            {
#if !defined(__CUDA_ARCH__)
                boost::ignore_unused(extents);
#endif
                return idx;
            }
        };
        //#############################################################################
        //! Maps a linear index to a 3 dimensional index.
        //#############################################################################
        template<>
        struct MapIdx<
            3u,
            1u>
        {
            //-----------------------------------------------------------------------------
            // \tparam TElem Type of the index values.
            // \param Index Idx to be mapped.
            // \param Extents Spatial size to map the index to.
            // \return Vector of dimension TuiDimDst.
            //-----------------------------------------------------------------------------
            template<
                typename TElem>
            ALPAKA_FN_HOST_ACC static auto mapIdx(
                Vec<dim::Dim<1u>, TElem> const & idx,
                Vec<dim::Dim<3u>, TElem> const & extents)
            -> Vec<dim::Dim<3u>, TElem>
            {
                auto const & uiIdx(idx[0]);
                auto const uiXyExtentsProd(extents.prod());
                auto const & uiExtentX(extents[2]);

                return {
                    uiIdx / uiXyExtentsProd,
                    (uiIdx % uiXyExtentsProd) / uiExtentX,
                    uiIdx % uiExtentX};
            }
        };
        //#############################################################################
        //! Maps a linear index to a 2 dimensional index.
        //#############################################################################
        template<>
        struct MapIdx<
            2u,
            1u>
        {
            //-----------------------------------------------------------------------------
            // \tparam TElem Type of the index values.
            // \param Index Idx to be mapped.
            // \param Extents Spatial size to map the index to.
            // \return Vector of dimension TuiDimDst.
            //-----------------------------------------------------------------------------
            template<
                typename TElem>
            ALPAKA_FN_HOST_ACC static auto mapIdx(
                Vec<dim::Dim<1u>, TElem> const & idx,
                Vec<dim::Dim<2u>, TElem> const & extents)
            -> Vec<dim::Dim<2u>, TElem>
            {
                auto const & uiIdx(idx[0]);
                auto const & uiExtentX(extents[1]);

                return {
                    uiIdx / uiExtentX,
                    uiIdx % uiExtentX};
            }
        };
        //#############################################################################
        //! Maps a 3 dimensional index to a linear index.
        //#############################################################################
        template<>
        struct MapIdx<
            1u,
            3u>
        {
            //-----------------------------------------------------------------------------
            // \tparam TElem Type of the index values.
            // \param Index Idx to be mapped.
            // \param Extents Spatial size to map the index to.
            // \return Vector of dimension TuiDimDst.
            //-----------------------------------------------------------------------------
            template<
                typename TElem>
            ALPAKA_FN_HOST_ACC static auto mapIdx(
                Vec<dim::Dim<3u>, TElem> const & idx,
                Vec<dim::Dim<3u>, TElem> const & extents)
            -> Vec<dim::Dim<1u>, TElem>
            {
                return (idx[0u] * extents[1u] + idx[1u]) * extents[2u] + idx[2u];
            }
        };
        //#############################################################################
        //! Maps a 2 dimensional index to a linear index.
        //#############################################################################
        template<>
        struct MapIdx<
            1u,
            2u>
        {
            //-----------------------------------------------------------------------------
            // \tparam TElem Type of the index values.
            // \param Index Idx to be mapped.
            // \param Extents Spatial size to map the index to.
            // \return Vector of dimension TuiDimDst.
            //-----------------------------------------------------------------------------
            template<
                typename TElem>
            ALPAKA_FN_HOST_ACC static auto mapIdx(
                Vec<dim::Dim<2u>, TElem> const & idx,
                Vec<dim::Dim<2u>, TElem> const & extents)
            -> Vec<dim::Dim<1u>, TElem>
            {
                return idx[0u] * extents[1u] + idx[1u];
            }
        };
    }

    //#############################################################################
    //! Maps a N dimensional index to a N dimensional position.
    //!
    //! \tparam TuiIdxDimOut Dimension of the index vector to map to.
    //! \tparam TuiIdxDimIn Dimension of the index vector to map from.
    //! \tparam TuiIdxDimExt Dimension of the extents vector to map use for mapping.
    //! \tparam TElem Type of the elements of the index vector to map from.
    //#############################################################################
    template<
        std::size_t TuiIdxDimOut,
        std::size_t TuiIdxDimIn,
        typename TElem>
    ALPAKA_FN_HOST_ACC auto mapIdx(
        Vec<dim::Dim<TuiIdxDimIn>, TElem> const & idx,
        Vec<dim::Dim<(TuiIdxDimOut < TuiIdxDimIn) ? TuiIdxDimIn : TuiIdxDimOut>, TElem> const & extents)
    -> Vec<dim::Dim<TuiIdxDimOut>, TElem>
    {
        return detail::MapIdx<
            TuiIdxDimOut,
            TuiIdxDimIn>
        ::mapIdx(
            idx,
            extents);
    }
}
