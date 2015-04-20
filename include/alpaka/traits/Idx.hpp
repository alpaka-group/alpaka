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
#include <alpaka/core/Vec.hpp>              // DimToVecT
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
                typename TDim>
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
            typename TDim = dim::Dim3,
            typename TIdx = void,
            typename TWorkDiv = void>
        ALPAKA_FCT_ACC auto getIdx(
            TIdx const & index,
            TWorkDiv const & workDiv)
        -> DimToVecT<TDim>
        {
            return traits::idx::GetIdx<
                TIdx,
                TOrigin,
                TUnit,
                TDim>
            ::getIdx(
                index,
                workDiv);
        }
    }

    namespace traits
    {
        namespace idx
        {
            //#############################################################################
            //! The 1D block thread index get trait specialization.
            //#############################################################################
            template<
                typename TIdx>
            struct GetIdx<
                TIdx,
                origin::Block,
                unit::Threads,
                alpaka::dim::Dim1>
            {
                //-----------------------------------------------------------------------------
                //! \return The linearized index of the current thread in the block.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_ACC static auto getIdx(
                    TIdx const & index,
                    TWorkDiv const & workDiv)
                -> alpaka::DimToVecT<alpaka::dim::Dim1>
                {
                    auto const v3uiBlockThreadExtents(alpaka::workdiv::getWorkDiv<origin::Block, unit::Threads, alpaka::dim::Dim3>(workDiv));
                    auto const v3uiBlockThreadIdx(alpaka::idx::getIdx<origin::Block, unit::Threads, alpaka::dim::Dim3>(index, workDiv));
                    return v3uiBlockThreadIdx[2] * v3uiBlockThreadExtents[1] * v3uiBlockThreadExtents[0] + v3uiBlockThreadIdx[1] * v3uiBlockThreadExtents[0] + v3uiBlockThreadIdx[0];
                }
            };
            //#############################################################################
            //! The 3D grid thread index get trait specialization.
            //#############################################################################
            template<
                typename TIdx>
            struct GetIdx<
                TIdx,
                origin::Grid,
                unit::Threads,
                alpaka::dim::Dim3>
            {
                //-----------------------------------------------------------------------------
                //! \return The 3-dimensional index of the current thread in grid.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_ACC static auto getIdx(
                    TIdx const & index,
                    TWorkDiv const & workDiv)
                -> alpaka::DimToVecT<alpaka::dim::Dim3>
                {
                    return alpaka::idx::getIdx<origin::Grid, unit::Blocks, alpaka::dim::Dim3>(index, workDiv)
                        * alpaka::workdiv::getWorkDiv<origin::Block, unit::Threads, alpaka::dim::Dim3>(workDiv)
                        + alpaka::idx::getIdx<origin::Block, unit::Threads, alpaka::dim::Dim3>(index, workDiv);
                }
            };
            //#############################################################################
            //! The 1D grid thread index get trait specialization.
            //#############################################################################
            template<
                typename TIdx>
            struct GetIdx<
                TIdx,
                origin::Grid,
                unit::Threads,
                alpaka::dim::Dim1>
            {
                //-----------------------------------------------------------------------------
                //! \return The linearized index of the current thread in the grid.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_ACC static auto getIdx(
                    TIdx const & index,
                    TWorkDiv const & workDiv)
                -> alpaka::DimToVecT<alpaka::dim::Dim1>
                {
                    auto const v3uiGridThreadSize(alpaka::workdiv::getWorkDiv<origin::Grid, unit::Threads, alpaka::dim::Dim3>(workDiv));
                    auto const v3uiGridThreadIdx(alpaka::idx::getIdx<origin::Grid, unit::Threads, alpaka::dim::Dim3>(index, workDiv));
                    return v3uiGridThreadIdx[2] * v3uiGridThreadSize[1] * v3uiGridThreadSize[0] + v3uiGridThreadIdx[1] * v3uiGridThreadSize[0] + v3uiGridThreadIdx[0];
                }
            };
            //#############################################################################
            //! The 1D grid block index get trait specialization.
            //#############################################################################
            template<
                typename TIdx>
            struct GetIdx<
                TIdx,
                origin::Grid,
                unit::Blocks,
                alpaka::dim::Dim1>
            {
                //-----------------------------------------------------------------------------
                //! \return The linearized index of the current block in the grid.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_ACC static auto getIdx(
                    TIdx const & index,
                    TWorkDiv const & workDiv)
                -> alpaka::DimToVecT<alpaka::dim::Dim1>
                {
                    auto const v3uiGridBlockExtent(alpaka::workdiv::getWorkDiv<origin::Grid, unit::Blocks, alpaka::dim::Dim3>(workDiv));
                    auto const v3uiGridBlockIdx(alpaka::idx::getIdx<origin::Grid, unit::Blocks, alpaka::dim::Dim3>(index, workDiv));
                    return v3uiGridBlockIdx[2] * v3uiGridBlockExtent[1] * v3uiGridBlockExtent[0] + v3uiGridBlockIdx[1] * v3uiGridBlockExtent[0] + v3uiGridBlockIdx[0];
                }
            };
        }
    }
}
