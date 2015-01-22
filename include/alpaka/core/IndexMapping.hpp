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

#include <alpaka/core/Common.hpp>   // ALPAKA_FCT_HOST
#include <alpaka/core/Vec.hpp>      // alpaka::Vec

#include <cstddef>                  // std::size_t

namespace alpaka
{
    namespace detail
    {
        //#############################################################################
        //! Maps a linear index to a N dimensional index.
        //!
        //! \tparam TuiDim dimension of the position to map to.
        //#############################################################################
        template<
            std::size_t TuiIndexDimDst, 
            std::size_t TuiIndexDimSrc>
        struct MapIndex;
        //#############################################################################
        //! Maps a linear index to a 3 dimensional index.
        //#############################################################################
        template<>
        struct MapIndex<
            3u, 
            1u>
        {
            //-----------------------------------------------------------------------------
            // \tparam TElem Type of the index values.
            // \param index Index to be mapped.
            // \param extents Spatial size to map the index to.
            // \return Vector of dimension TuiDimDst.
            //-----------------------------------------------------------------------------
            template<
                typename TElem>
            ALPAKA_FCT_HOST_ACC static Vec<3u> mapIndex(
                Vec<1u, TElem> const & index, 
                Vec<2u, TElem> const & extents)
            {
                auto const & uiIndex(index[0]);
                auto const uiExtentXyLin(extents.prod());
                auto const & uiExtentX(extents[0]);

                return {
                    uiIndex % uiExtentX,
                    (uiIndex % uiExtentXyLin) / uiExtentX,
                    uiIndex / uiExtentXyLin
                };
            }
        };
        //#############################################################################
        //! Maps a linear index to a 2 dimensional index.
        //#############################################################################
        template<>
        struct MapIndex<
            2u, 
            1u>
        {
            //-----------------------------------------------------------------------------
            // \tparam TElem Type of the index values.
            // \param index Index to be mapped.
            // \param extents Spatial size to map the index to.
            // \return Vector of dimension TuiDimDst.
            //-----------------------------------------------------------------------------
            template<
                typename TElem>
            ALPAKA_FCT_HOST_ACC static Vec<2u> mapIndex(
                Vec<1u, TElem> const & index, 
                Vec<1u, TElem> const & extents)
            {
                auto const & uiIndex(index[0]);
                auto const & uiExtentX(extents[0]);

                return {
                    uiIndex % uiExtentX,
                    uiIndex / uiExtentX
                };
            }
        };
    }

    //#############################################################################
    //! Maps a N dimensional index to a N dimensional position.
    //!
    //! \tparam TuiDim dimension of the position to map to.
    //! \tparam TuiIndexDim dimension of the index vector to map from.
    //! \tparam TElem type of the elements of the index vector to map from.
    //#############################################################################
    template<
        std::size_t TuiIndexDimDst, 
        std::size_t TuiIndexDimSrc, 
        typename TElem>
    Vec<TuiIndexDimDst> mapIndex(
        Vec<TuiIndexDimSrc, TElem> const & index, 
        Vec<TuiIndexDimDst-1u, TElem> const & extents)
    {
        return detail::MapIndex<TuiIndexDimDst, TuiIndexDimSrc>::mapIndex(index, extents);
    }
}
