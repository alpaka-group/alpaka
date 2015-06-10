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

#include <alpaka/traits/Dim.hpp>            // Dim
#include <alpaka/traits/WorkDiv.hpp>        // workdiv::getWorkDiv
#include <alpaka/traits/Idx.hpp>            // idx::getIdx

#include <alpaka/core/BasicDims.hpp>        // dim::Dim<N>
#include <alpaka/core/Positioning.hpp>      // origin::Grid/Blocks, unit::Blocks, unit::Threads
#include <alpaka/core/Vec.hpp>              // Vec<N>
#include <alpaka/core/Common.hpp>           // ALPAKA_FCT_ACC

#include <utility>                          // std::forward

namespace alpaka
{
    namespace traits
    {
        //-----------------------------------------------------------------------------
        //! The index traits.
        //-----------------------------------------------------------------------------
        namespace idx
        {
            //#############################################################################
            //! The index get trait.
            //#############################################################################
            template<
                typename TIdx,
                typename TOrigin,
                typename TUnit,
                typename TSfinae = void>
            struct GetIdx;
        }
    }

    //-----------------------------------------------------------------------------
    //! The index traits accessors.
    //-----------------------------------------------------------------------------
    namespace idx
    {
        //-----------------------------------------------------------------------------
        //! Get the indices requested.
        //-----------------------------------------------------------------------------
        template<
            typename TOrigin,
            typename TUnit,
            typename TIdx,
            typename TWorkDiv>
        ALPAKA_FCT_ACC auto getIdx(
            TIdx const & idx,
            TWorkDiv const & workDiv)
        -> Vec<alpaka::dim::DimT<TWorkDiv>>
        {
            return traits::idx::GetIdx<
                TIdx,
                TOrigin,
                TUnit>
            ::getIdx(
                idx,
                workDiv);
        }
    }

    namespace traits
    {
        namespace idx
        {
            //#############################################################################
            //! The grid thread index get trait specialization.
            //#############################################################################
            template<
                typename TIdx>
            struct GetIdx<
                TIdx,
                origin::Grid,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current thread in the grid.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_ACC static auto getIdx(
                    TIdx const & idx,
                    TWorkDiv const & workDiv)
                -> alpaka::Vec<alpaka::dim::DimT<TWorkDiv>>
                {
                    return alpaka::idx::getIdx<origin::Grid, unit::Blocks>(idx, workDiv)
                        * alpaka::workdiv::getWorkDiv<origin::Block, unit::Threads>(workDiv)
                        + alpaka::idx::getIdx<origin::Block, unit::Threads>(idx, workDiv);
                }
            };
        }
    }
}
