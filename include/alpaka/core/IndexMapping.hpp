/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of either the GNU General Public License or
* the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License and the GNU Lesser General Public License
* for more details.
*
* You should have received a copy of the GNU General Public License
* and the GNU Lesser General Public License along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/core/Vec.hpp>  // alpaka::vec

#include <cstddef>              // std::size_t

namespace alpaka
{
    namespace detail
    {
        //#############################################################################
        //! Maps a linear index to a N dimensional position
        //!
        //! \tparam TuiDim dimension of the position to map to.
        //#############################################################################
        template<std::size_t TuiIndexDimDst, std::size_t TuiIndexDimSrc>
        struct MapIndex;

        template<>
        struct MapIndex<3u, 1u>
        {
            //-----------------------------------------------------------------------------
            // \tparam TIndex type of the index values
            // \param index Index to be mapped.
            // \param extent Spatial size to map the index to.
            // \return Vector of dimension TuiDimDst.
            //-----------------------------------------------------------------------------
            template<typename TIndex>
            ALPAKA_FCT_HOST_ACC vec<3u> operator()(vec<1u, TIndex> const & index, vec<2u, TIndex> const & extent) const
            {
                auto const & uiIndex(index[0]);
                auto const uiExtentXyLin(extent.prod());
                auto const & uiExtentX(extent[0]);

                return {
                    uiIndex % uiExtentX,
                    (uiIndex % uiExtentXyLin) / uiExtentX,
                    uiIndex / uiExtentXyLin
                };
            }
        };

        template<>
        struct MapIndex<2u, 1u>
        {
            //-----------------------------------------------------------------------------
            // \tparam TIndex type of the index values
            // \param index Index to be mapped.
            // \param extent Spatial size to map the index to.
            // \return Vector of dimension TuiDimDst.
            //-----------------------------------------------------------------------------
            template<typename TIndex>
            ALPAKA_FCT_HOST_ACC vec<3u> operator()(vec<1u, TIndex> const & index, vec<1u, TIndex> const & extent) const
            {
                auto const & uiIndex(index[0]);
                auto const & uiExtentX(extent[0]);

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
    //! \tparam TIndex type of the index vector to map from.
    //#############################################################################
    template<std::size_t TuiIndexDimDst, std::size_t TuiIndexDimSrc, typename TIndex>
    auto mapIndex(vec<TuiIndexDimSrc, TIndex> const & index, vec<TuiIndexDimDst-1u, TIndex> const & extent)
        -> typename std::result_of<detail::MapIndex<TuiIndexDimDst, TuiIndexDimSrc>(vec<TuiIndexDimSrc, TIndex>, vec<TuiIndexDimDst-1u, TIndex>)>::type
    {
        return detail::MapIndex<TuiIndexDimDst, TuiIndexDimSrc>()(index, extent);
    }
}
