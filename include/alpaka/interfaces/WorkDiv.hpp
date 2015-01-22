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

#include <alpaka/traits/Dim.hpp>        // alpaka::dim::DimToVecT

#include <alpaka/core/BasicDims.hpp>    // alpaka::dim::Dim<N>
#include <alpaka/core/Positioning.hpp>  // alpaka::origin::Grid/Blocks, alpaka::unit::Blocks, alpaka::unit::Kernels
#include <alpaka/core/Common.hpp>       // ALPAKA_FCT_ACC

#include <utility>                      // std::forward

namespace alpaka
{
    namespace detail
    {
        //#############################################################################
        //! The extents trait.
        //#############################################################################
        template<
            typename TWorkDiv, 
            typename TOrigin, 
            typename TUnit, 
            typename TDimensionality>
        struct GetExtents;

        //#############################################################################
        //! The 3D block kernels extent trait specialization.
        //#############################################################################
        template<
            typename TWorkDiv>
        struct GetExtents<
            TWorkDiv, 
            origin::Block, 
            unit::Kernels, 
            dim::Dim3>
        {
            //-----------------------------------------------------------------------------
            //! \return The number of kernels in each dimension of a block.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST_ACC static dim::DimToVecT<dim::Dim3> getExtents(
                TWorkDiv const & workDiv)
            {
                return workDiv.getBlockKernelsExtents();
            }
        };
        //#############################################################################
        //! The 1D block kernels extent trait specialization.
        //#############################################################################
        template<
            typename TWorkDiv>
        struct GetExtents<
            TWorkDiv, 
            origin::Block, 
            unit::Kernels, 
            dim::Dim1>
        {
            //-----------------------------------------------------------------------------
            //! \return The number of kernels in a block.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST_ACC static dim::DimToVecT<dim::Dim1> getExtents(
                TWorkDiv const & workDiv)
            {
                return GetExtents<TWorkDiv, origin::Block, unit::Kernels, dim::Dim3>::getExtents(workDiv).prod();
            }
        };
        //#############################################################################
        //! The 3D grif kernels extent trait specialization.
        //#############################################################################
        template<
            typename TWorkDiv>
        struct GetExtents<
            TWorkDiv, 
            origin::Grid, 
            unit::Kernels, 
            dim::Dim3>
        {
            //-----------------------------------------------------------------------------
            //! \return The number of kernels in each dimension of the grid.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST_ACC static dim::DimToVecT<dim::Dim3> getExtents(
                TWorkDiv const & workDiv)
            {
                return workDiv.getGridBlocksExtents() * workDiv.getBlockKernelsExtents();
            }
        };
        //#############################################################################
        //! The 1D grid kernels extent trait specialization.
        //#############################################################################
        template<
            typename TWorkDiv>
        struct GetExtents<
            TWorkDiv, 
            origin::Grid, 
            unit::Kernels, 
            dim::Dim1>
        {
            //-----------------------------------------------------------------------------
            //! \return The number of kernels in the grid.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST_ACC static dim::DimToVecT<dim::Dim1> getExtents(
                TWorkDiv const & workDiv)
            {
                return GetExtents<TWorkDiv, origin::Grid, unit::Kernels, dim::Dim3>::getExtents(workDiv).prod();
            }
        };
        //#############################################################################
        //! The 3D grid blocks extent trait specialization.
        //#############################################################################
        template<
            typename TWorkDiv>
        struct GetExtents<
            TWorkDiv, 
            origin::Grid, 
            unit::Blocks, 
            dim::Dim3>
        {
            //-----------------------------------------------------------------------------
            //! \return The number of blocks in each dimension of the grid.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST_ACC static dim::DimToVecT<dim::Dim3> getExtents(
                TWorkDiv const & workDiv)
            {
                return workDiv.getGridBlocksExtents();
            }
        };
        //#############################################################################
        //! The 1D grid blocks extent trait specialization.
        //#############################################################################
        template<
            typename TWorkDiv>
        struct GetExtents<
            TWorkDiv, 
            origin::Grid, 
            unit::Blocks, 
            dim::Dim1>
        {
            //-----------------------------------------------------------------------------
            //! \return The number of blocks in the grid.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST_ACC static dim::DimToVecT<dim::Dim1> getExtents(
                TWorkDiv const & workDiv)
            {
                return GetExtents<TWorkDiv, origin::Grid, unit::Blocks, dim::Dim3>::getExtents(workDiv).prod();
            }
        };
    }

    //#############################################################################
    //! The work extent interface.
    //#############################################################################
    template<
        typename TWorkDiv>
    class IWorkDiv :
        private TWorkDiv
    {
        //-----------------------------------------------------------------------------
        //! Stream out operator.
        //-----------------------------------------------------------------------------
        template<
            typename TWorkDiv2>
        friend std::ostream & operator << (
            std::ostream & os, 
            IWorkDiv<TWorkDiv2> const & workDiv);

    public:
        //-----------------------------------------------------------------------------
        //! Constructor.
        //-----------------------------------------------------------------------------
        template<
            typename... TArgs>
        ALPAKA_FCT_HOST_ACC IWorkDiv(
            TArgs && ... args) :
            TWorkDiv(std::forward<TArgs>(args)...)
        {}

        //-----------------------------------------------------------------------------
        //! Get the size requested.
        //-----------------------------------------------------------------------------
        template<
            typename TOrigin, 
            typename TUnit,
            typename TDimensionality = dim::Dim3>
        ALPAKA_FCT_HOST_ACC typename dim::DimToVecT<TDimensionality> getExtents() const
        {
            return alpaka::detail::GetExtents<TWorkDiv, TOrigin, TUnit, TDimensionality>::getExtents(
                *static_cast<TWorkDiv const *>(this));
        }
    };

    //-----------------------------------------------------------------------------
    //! Stream out operator.
    //-----------------------------------------------------------------------------
    template<
        typename TWorkDiv>
    ALPAKA_FCT_HOST std::ostream & operator << (
        std::ostream & os, 
        IWorkDiv<TWorkDiv> const & workDiv)
    {
        return (os << "{GridBlocks: " << workDiv.getGridBlocksExtents() << ", BlockKernels: " << workDiv.getBlockKernelsExtents() << "}");
    }
}
