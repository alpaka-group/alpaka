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

#include <alpaka/traits/Dim.hpp>        // dim::DimToVecT

#include <alpaka/core/BasicDims.hpp>    // dim::Dim<N>
#include <alpaka/core/Positioning.hpp>  // origin::Grid/Blocks, unit::Blocks, unit::Kernels
#include <alpaka/core/Common.hpp>       // ALPAKA_FCT_ACC

#include <utility>                      // std::forward

namespace alpaka
{
    namespace traits
    {
        //-----------------------------------------------------------------------------
        //! The work division traits.
        //-----------------------------------------------------------------------------
        namespace workdiv
        {
            //#############################################################################
            //! The work div trait.
            //#############################################################################
            template<
                typename TWorkDiv,
                typename TOrigin,
                typename TUnit,
                typename TDimensionality,
                typename TSfinae = void>
            struct GetWorkDiv;
        }
    }

    namespace traits
    {
        namespace workdiv
        {
            //#############################################################################
            //! The work div block kernels 1D extents trait specialization.
            //#############################################################################
            template<
                typename TWorkDiv>
            struct GetWorkDiv<
                TWorkDiv,
                origin::Block,
                unit::Kernels,
                alpaka::dim::Dim1>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of kernels in a block.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST_ACC static alpaka::dim::DimToVecT<alpaka::dim::Dim1> getWorkDiv(
                    TWorkDiv const & workDiv)
                {
                    return GetWorkDiv<TWorkDiv, origin::Block, unit::Kernels, alpaka::dim::Dim3>::getWorkDiv(workDiv).prod();
                }
            };
            //#############################################################################
            //! The work div grid kernels 3D extents trait specialization.
            //#############################################################################
            template<
                typename TWorkDiv>
            struct GetWorkDiv<
                TWorkDiv,
                origin::Grid,
                unit::Kernels,
                alpaka::dim::Dim3>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of kernels in each dimension of the grid.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST_ACC static alpaka::dim::DimToVecT<alpaka::dim::Dim3> getWorkDiv(
                    TWorkDiv const & workDiv)
                {
                    return GetWorkDiv<TWorkDiv, origin::Grid, unit::Blocks, alpaka::dim::Dim3>::getWorkDiv(workDiv)
                        * GetWorkDiv<TWorkDiv, origin::Block, unit::Kernels, alpaka::dim::Dim3>::getWorkDiv(workDiv);
                }
            };
            //#############################################################################
            //! The work div grid kernels 1D extents trait specialization.
            //#############################################################################
            template<
                typename TWorkDiv>
            struct GetWorkDiv<
                TWorkDiv,
                origin::Grid,
                unit::Kernels,
                alpaka::dim::Dim1>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of kernels in the grid.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST_ACC static alpaka::dim::DimToVecT<alpaka::dim::Dim1> getWorkDiv(
                    TWorkDiv const & workDiv)
                {
                    return GetWorkDiv<TWorkDiv, origin::Grid, unit::Kernels, alpaka::dim::Dim3>::getWorkDiv(workDiv).prod();
                }
            };
            //#############################################################################
            //! The work div grid blocks 1D extents trait specialization.
            //#############################################################################
            template<
                typename TWorkDiv>
            struct GetWorkDiv<
                TWorkDiv,
                origin::Grid,
                unit::Blocks,
                alpaka::dim::Dim1>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of blocks in the grid.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST_ACC static alpaka::dim::DimToVecT<alpaka::dim::Dim1> getWorkDiv(
                    TWorkDiv const & workDiv)
                {
                    return GetWorkDiv<TWorkDiv, origin::Grid, unit::Blocks, alpaka::dim::Dim3>::getWorkDiv(workDiv).prod();
                }
            };
        }
    }

    //-----------------------------------------------------------------------------
    //! The work division traits accessors.
    //-----------------------------------------------------------------------------
    namespace workdiv
    {
        //-----------------------------------------------------------------------------
        //! Get the extents requested.
        //-----------------------------------------------------------------------------
        template<
            typename TOrigin,
            typename TUnit,
            typename TDimensionality = dim::Dim3,
            typename TWorkDiv = void>
        ALPAKA_FCT_HOST_ACC typename dim::DimToVecT<TDimensionality> getWorkDiv(
            TWorkDiv const & workDiv)
        {
            return traits::workdiv::GetWorkDiv<TWorkDiv, TOrigin, TUnit, TDimensionality>::getWorkDiv(
                workDiv);
        }
    }
}
