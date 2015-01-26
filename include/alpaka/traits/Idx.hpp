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

#include <alpaka/traits/Dim.hpp>            // dim::DimToVecT
#include <alpaka/traits/WorkDiv.hpp>        // workdiv::getWorkDiv
#include <alpaka/traits/Idx.hpp>            // idx::getIdx

#include <alpaka/core/BasicDims.hpp>        // dim::Dim<N>
#include <alpaka/core/Positioning.hpp>      // origin::Grid/Blocks, unit::Blocks, unit::Kernels
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
                typename TDimensionality>
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
            typename TDimensionality = dim::Dim3,
            typename TIdx = void,
            typename TWorkDiv = void>
        ALPAKA_FCT_ACC typename dim::DimToVecT<TDimensionality> getIdx(
            TIdx const & index,
            TWorkDiv const & workDiv)
        {
            return traits::idx::GetIdx<TIdx, TOrigin, TUnit, TDimensionality>::getIdx(
                index,
                workDiv);
        }
    }

    namespace traits
    {
        namespace idx
        {
            //#############################################################################
            //! The 1D block kernels index get trait specialization.
            //#############################################################################
            template<
                typename TIdx>
            struct GetIdx<
                TIdx,
                origin::Block,
                unit::Kernels,
                alpaka::dim::Dim1>
            {
                //-----------------------------------------------------------------------------
                //! \return The linearized index of the current kernel in the block.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_ACC static alpaka::dim::DimToVecT<alpaka::dim::Dim1> getIdx(
                    TIdx const & index,
                    TWorkDiv const & workDiv)
                {
                    auto const v3uiBlockKernelsExtents(alpaka::workdiv::getWorkDiv<origin::Block, unit::Kernels, alpaka::dim::Dim3>(workDiv));
                    auto const v3uiBlockKernelIdx(alpaka::idx::getIdx<origin::Block, unit::Kernels, alpaka::dim::Dim3>(index, workDiv));
                    return v3uiBlockKernelIdx[2] * v3uiBlockKernelsExtents[1] * v3uiBlockKernelsExtents[0] + v3uiBlockKernelIdx[1] * v3uiBlockKernelsExtents[0] + v3uiBlockKernelIdx[0];
                }
            };
            //#############################################################################
            //! The 3D grid kernels index get trait specialization.
            //#############################################################################
            template<
                typename TIdx>
            struct GetIdx<
                TIdx,
                origin::Grid,
                unit::Kernels,
                alpaka::dim::Dim3>
            {
                //-----------------------------------------------------------------------------
                //! \return The 3-dimensional index of the current kernel in grid.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_ACC static alpaka::dim::DimToVecT<alpaka::dim::Dim3> getIdx(
                    TIdx const & index,
                    TWorkDiv const & workDiv)
                {
                    return alpaka::idx::getIdx<origin::Grid, unit::Blocks, alpaka::dim::Dim3>(index, workDiv)
                        * alpaka::workdiv::getWorkDiv<origin::Block, unit::Kernels, alpaka::dim::Dim3>(workDiv)
                        + alpaka::idx::getIdx<origin::Block, unit::Kernels, alpaka::dim::Dim3>(index, workDiv);
                }
            };
            //#############################################################################
            //! The 1D grid kernels index get trait specialization.
            //#############################################################################
            template<
                typename TIdx>
            struct GetIdx<
                TIdx,
                origin::Grid,
                unit::Kernels,
                alpaka::dim::Dim1>
            {
                //-----------------------------------------------------------------------------
                //! \return The linearized index of the current kernel in the grid.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_ACC static alpaka::dim::DimToVecT<alpaka::dim::Dim1> getIdx(
                    TIdx const & index,
                    TWorkDiv const & workDiv)
                {
                    auto const v3uiGridKernelSize(alpaka::workdiv::getWorkDiv<origin::Grid, unit::Kernels, alpaka::dim::Dim3>(workDiv));
                    auto const v3uiGridKernelIdx(alpaka::idx::getIdx<origin::Grid, unit::Kernels, alpaka::dim::Dim3>(index, workDiv));
                    return v3uiGridKernelIdx[2] * v3uiGridKernelSize[1] * v3uiGridKernelSize[0] + v3uiGridKernelIdx[1] * v3uiGridKernelSize[0] + v3uiGridKernelIdx[0];
                }
            };
            //#############################################################################
            //! The 1D grid blocks index get trait specialization.
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
                ALPAKA_FCT_ACC static alpaka::dim::DimToVecT<alpaka::dim::Dim1> getIdx(
                    TIdx const & index,
                    TWorkDiv const & workDiv)
                {
                    auto const v3uiGridBlocksExtent(alpaka::workdiv::getWorkDiv<origin::Grid, unit::Blocks, alpaka::dim::Dim3>(workDiv));
                    auto const v3uiGridBlockIdx(alpaka::idx::getIdx<origin::Grid, unit::Blocks, alpaka::dim::Dim3>(index, workDiv));
                    return v3uiGridBlockIdx[2] * v3uiGridBlocksExtent[1] * v3uiGridBlocksExtent[0] + v3uiGridBlockIdx[1] * v3uiGridBlocksExtent[0] + v3uiGridBlockIdx[0];
                }
            };
        }
    }
}
