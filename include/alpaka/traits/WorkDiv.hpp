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

#include <alpaka/traits/Dim.hpp>        // Dim

#include <alpaka/core/BasicDims.hpp>    // dim::Dim<N>
#include <alpaka/core/Vec.hpp>          // Vec<N>, DimToVecT
#include <alpaka/core/Positioning.hpp>  // origin::Grid/Blocks, unit::Blocks, unit::Threads
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
                typename TDim,
                typename TSfinae = void>
            struct GetWorkDiv;
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
            typename TDim = dim::Dim3,
            typename TWorkDiv = void>
        ALPAKA_FCT_HOST_ACC DimToVecT<TDim> getWorkDiv(
            TWorkDiv const & workDiv)
        {
            return traits::workdiv::GetWorkDiv<TWorkDiv, TOrigin, TUnit, TDim>::getWorkDiv(
                workDiv);
        }
    }

    namespace traits
    {
        namespace workdiv
        {
            //#############################################################################
            //! The work div block threads 1D extents trait specialization.
            //#############################################################################
            template<
                typename TWorkDiv>
            struct GetWorkDiv<
                TWorkDiv,
                origin::Block,
                unit::Threads,
                alpaka::dim::Dim1>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of threads in a block.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST_ACC static alpaka::DimToVecT<alpaka::dim::Dim1> getWorkDiv(
                    TWorkDiv const & workDiv)
                {
                    return alpaka::workdiv::getWorkDiv<origin::Block, unit::Threads, alpaka::dim::Dim3>(workDiv).prod();
                }
            };
            //#############################################################################
            //! The work div grid threads 3D extents trait specialization.
            //#############################################################################
            template<
                typename TWorkDiv>
            struct GetWorkDiv<
                TWorkDiv,
                origin::Grid,
                unit::Threads,
                alpaka::dim::Dim3>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of threads in each dimension of the grid.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST_ACC static alpaka::DimToVecT<alpaka::dim::Dim3> getWorkDiv(
                    TWorkDiv const & workDiv)
                {
                    return alpaka::workdiv::getWorkDiv<origin::Grid, unit::Blocks, alpaka::dim::Dim3>(workDiv)
                        * alpaka::workdiv::getWorkDiv<origin::Block, unit::Threads, alpaka::dim::Dim3>(workDiv);
                }
            };
            //#############################################################################
            //! The work div grid threads 1D extents trait specialization.
            //#############################################################################
            template<
                typename TWorkDiv>
            struct GetWorkDiv<
                TWorkDiv,
                origin::Grid,
                unit::Threads,
                alpaka::dim::Dim1>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of threads in the grid.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST_ACC static alpaka::DimToVecT<alpaka::dim::Dim1> getWorkDiv(
                    TWorkDiv const & workDiv)
                {
                    return alpaka::workdiv::getWorkDiv<origin::Grid, unit::Threads, alpaka::dim::Dim3>(workDiv).prod();
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
                ALPAKA_FCT_HOST_ACC static alpaka::DimToVecT<alpaka::dim::Dim1> getWorkDiv(
                    TWorkDiv const & workDiv)
                {
                    return alpaka::workdiv::getWorkDiv<origin::Grid, unit::Blocks, alpaka::dim::Dim3>(workDiv).prod();
                }
            };
        }
    }
}
