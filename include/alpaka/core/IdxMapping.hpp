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

#include <alpaka/core/Vec.hpp>      // Vec
#include <alpaka/core/Common.hpp>   // ALPAKA_FCT_HOST_ACC

namespace alpaka
{
    namespace detail
    {
        //#############################################################################
        //! Maps a linear index to a N dimensional index.
        //#############################################################################
        template<
            UInt TuiIdxDimDst,
            UInt TuiIdxDimSrc>
        struct MapIdx;
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
            // \param index Idx to be mapped.
            // \param extents Spatial size to map the index to.
            // \return Vector of dimension TuiDimDst.
            //-----------------------------------------------------------------------------
            template<
                typename TElem>
            ALPAKA_FCT_HOST_ACC static auto mapIdx(
                Vec<dim::Dim1, TElem> const & index,
                Vec<dim::Dim2, TElem> const & extents)
            -> Vec<dim::Dim3, TElem>
            {
                auto const & uiIdx(index[0]);
                auto const uiExtentXyLin(extents.prod());
                auto const & uiExtentX(extents[0]);

                return {
                    uiIdx % uiExtentX,
                    (uiIdx % uiExtentXyLin) / uiExtentX,
                    uiIdx / uiExtentXyLin
                };
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
            // \param index Idx to be mapped.
            // \param extents Spatial size to map the index to.
            // \return Vector of dimension TuiDimDst.
            //-----------------------------------------------------------------------------
            template<
                typename TElem>
            ALPAKA_FCT_HOST_ACC static auto mapIdx(
                Vec<dim::Dim1, TElem> const & index,
                Vec<dim::Dim1, TElem> const & extents)
            -> Vec<dim::Dim2, TElem>
            {
                auto const & uiIdx(index[0]);
                auto const & uiExtentX(extents[0]);

                return {
                    uiIdx % uiExtentX,
                    uiIdx / uiExtentX
                };
            }
        };
    }

    //#############################################################################
    //! Maps a N dimensional index to a N dimensional position.
    //!
    //! \tparam TuiIdxDimDst dimension of the position to map to.
    //! \tparam TuiIdxDimSrc dimension of the index vector to map from.
    //! \tparam TElem type of the elements of the index vector to map from.
    //#############################################################################
    template<
        UInt TuiIdxDimDst,
        UInt TuiIdxDimSrc,
        typename TElem>
    ALPAKA_FCT_HOST_ACC auto mapIdx(
        Vec<dim::Dim<TuiIdxDimSrc>, TElem> const & index,
        Vec<dim::Dim<TuiIdxDimDst-1u>, TElem> const & extents)
    -> Vec<dim::Dim<TuiIdxDimDst>, TElem>
    {
        return detail::MapIdx<
            TuiIdxDimDst,
            TuiIdxDimSrc>
        ::mapIdx(
            index,
            extents);
    }
}
