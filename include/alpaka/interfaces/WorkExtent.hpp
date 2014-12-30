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

#include <alpaka/core/Positioning.hpp>  // alpaka::origin::Grid/Blocks
#include <alpaka/core/Common.hpp>       // ALPAKA_FCT_HOST_ACC

#include <utility>                      // std::forward

namespace alpaka
{
    namespace detail
    {
        //#############################################################################
        //! The template to get the requested size.
        //#############################################################################
        template<typename TWorkExtent, typename TOrigin, typename TUnit, typename TDimensionality>
        struct GetExtent;

        template<typename TWorkExtent>
        struct GetExtent<TWorkExtent, origin::Block, unit::Kernels, dim::D3>
        {
            //-----------------------------------------------------------------------------
            //! \return The number of kernels in each dimension of a block.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST_ACC static DimToRetType<dim::D3>::type getExtent(TWorkExtent const & workExtent)
            {
                return workExtent.getExtentBlockKernels();
            }
        };
        template<typename TWorkExtent>
        struct GetExtent<TWorkExtent, origin::Block, unit::Kernels, dim::Linear>
        {
            //-----------------------------------------------------------------------------
            //! \return The number of kernels in a block.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST_ACC static DimToRetType<dim::Linear>::type getExtent(TWorkExtent const & workExtent)
            {
                return GetExtent<TWorkExtent, origin::Block, unit::Kernels, dim::D3>::getExtent(workExtent).prod();
            }
        };
        template<typename TWorkExtent>
        struct GetExtent<TWorkExtent, origin::Grid, unit::Kernels, dim::D3>
        {
            //-----------------------------------------------------------------------------
            //! \return The number of kernels in each dimension of the grid.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST_ACC static DimToRetType<dim::D3>::type getExtent(TWorkExtent const & workExtent)
            {
                return workExtent.getExtentGridBlocks() * workExtent.getExtentBlockKernels();
            }
        };
        template<typename TWorkExtent>
        struct GetExtent<TWorkExtent, origin::Grid, unit::Kernels, dim::Linear>
        {
            //-----------------------------------------------------------------------------
            //! \return The number of kernels in the grid.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST_ACC static DimToRetType<dim::Linear>::type getExtent(TWorkExtent const & workExtent)
            {
                return GetExtent<TWorkExtent, origin::Grid, unit::Kernels, dim::D3>::getExtent(workExtent).prod();
            }
        };
        template<typename TWorkExtent>
        struct GetExtent<TWorkExtent, origin::Grid, unit::Blocks, dim::D3>
        {
            //-----------------------------------------------------------------------------
            //! \return The number of blocks in each dimension of the grid.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST_ACC static DimToRetType<dim::D3>::type getExtent(TWorkExtent const & workExtent)
            {
                return workExtent.getExtentGridBlocks();
            }
        };
        template<typename TWorkExtent>
        struct GetExtent<TWorkExtent, origin::Grid, unit::Blocks, dim::Linear>
        {
            //-----------------------------------------------------------------------------
            //! \return The number of blocks in the grid.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST_ACC static DimToRetType<dim::Linear>::type getExtent(TWorkExtent const & workExtent)
            {
                return GetExtent<TWorkExtent, origin::Grid, unit::Blocks, dim::D3>::getExtent(workExtent).prod();
            }
        };
    }

    //#############################################################################
    //! The work extent interface.
    //#############################################################################
    template<typename TWorkExtent>
    class IWorkExtent :
        private TWorkExtent
    {
        //-----------------------------------------------------------------------------
        //! Stream out operator.
        //-----------------------------------------------------------------------------
        template<typename TWorkExtent2>
        friend std::ostream & operator << (std::ostream & os, IWorkExtent<TWorkExtent2> const & workExtent);

    public:
        //-----------------------------------------------------------------------------
        //! Constructor.
        //-----------------------------------------------------------------------------
        template<typename... TArgs>
        ALPAKA_FCT_HOST_ACC IWorkExtent(TArgs && ... args) :
            TWorkExtent(std::forward<TArgs>(args)...)
        {}

        //-----------------------------------------------------------------------------
        //! Get the size requested.
        //-----------------------------------------------------------------------------
        template<typename TOrigin, typename TUnit, typename TDimensionality = dim::D3>
        ALPAKA_FCT_HOST_ACC typename detail::DimToRetType<TDimensionality>::type getExtent() const
        {
            return detail::GetExtent<TWorkExtent, TOrigin, TUnit, TDimensionality>::getExtent(
                *static_cast<TWorkExtent const *>(this));
        }
    };

    //-----------------------------------------------------------------------------
    //! Stream out operator.
    //-----------------------------------------------------------------------------
    template<typename TWorkExtent>
    ALPAKA_FCT_HOST std::ostream & operator << (std::ostream & os, IWorkExtent<TWorkExtent> const & workExtent)
    {
        return (os << "{GridBlocks: " << workExtent.getExtentGridBlocks() << ", BlockKernels: " << workExtent.getExtentBlockKernels() << "}");
    }
}
