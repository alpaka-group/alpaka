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

#include <alpaka/core/Positioning.hpp>      // alpaka::origin::Grid/Blocks
#include <alpaka/core/Common.hpp>           // ALPAKA_FCT_HOST_ACC

#include <alpaka/host/WorkExtent.hpp>       // alpaka::WorkExtentHost

#include <utility>                          // std::forward

namespace alpaka
{
    namespace detail
    {
        //#############################################################################
        //! The abstract index getter.
        //#############################################################################
        template<typename TIndex, typename TOrigin, typename TUnit, typename TDimensionality>
        struct GetIdx;

        template<typename TIndex>
        struct GetIdx<TIndex, origin::Block, unit::Kernels, dim::D3>
        {
            //-----------------------------------------------------------------------------
            //! \return The 3-dimensional index of the current kernel in the block.
            //-----------------------------------------------------------------------------
            template<typename TWorkExtent>
            ALPAKA_FCT_HOST_ACC static DimToRetType<dim::D3>::type getIdx(TIndex const & index, IWorkExtent<TWorkExtent> const &)
            {
                return index.getIdxBlockKernel();
            }
        };
        template<typename TIndex>
        struct GetIdx<TIndex, origin::Block, unit::Kernels, dim::Linear>
        {
            //-----------------------------------------------------------------------------
            //! \return The linearized index of the current kernel in the block.
            //-----------------------------------------------------------------------------
            template<typename TWorkExtent>
            ALPAKA_FCT_HOST_ACC static DimToRetType<dim::Linear>::type getIdx(TIndex const & index, IWorkExtent<TWorkExtent> const & workExtent)
            {
                auto const v3uiBlockKernelsExtent(workExtent.template getExtent<origin::Block, unit::Kernels, dim::D3>());
                auto const v3uiBlockKernelIdx(GetIdx<TIndex, origin::Block, unit::Kernels, dim::D3>::getIdx(index, workExtent));
                return v3uiBlockKernelIdx[2]*v3uiBlockKernelsExtent[1]*v3uiBlockKernelsExtent[0] + v3uiBlockKernelIdx[1]*v3uiBlockKernelsExtent[0] + v3uiBlockKernelIdx[0];
            }
        };
        template<typename TIndex>
        struct GetIdx<TIndex, origin::Grid, unit::Kernels, dim::D3>
        {
            //-----------------------------------------------------------------------------
            //! \return The 3-dimensional index of the current kernel in grid.
            //-----------------------------------------------------------------------------
            template<typename TWorkExtent>
            ALPAKA_FCT_HOST_ACC static DimToRetType<dim::D3>::type getIdx(TIndex const & index, IWorkExtent<TWorkExtent> const & workExtent)
            {
                return 
                    index.getIdxGridBlock() * workExtent.template getExtent<origin::Block, unit::Kernels, dim::D3>()
                    + index.getIdxBlockKernel();
            }
        };
        template<typename TIndex>
        struct GetIdx<TIndex, origin::Grid, unit::Kernels, dim::Linear>
        {
            //-----------------------------------------------------------------------------
            //! \return The linearized index of the current kernel in the grid.
            //-----------------------------------------------------------------------------
            template<typename TWorkExtent>
            ALPAKA_FCT_HOST_ACC static DimToRetType<dim::Linear>::type getIdx(TIndex const & index, IWorkExtent<TWorkExtent> const & workExtent)
            {
                auto const v3uiGridKernelSize(workExtent.template getExtent<origin::Grid, unit::Kernels, dim::D3>());
                auto const v3uiGridKernelIdx(GetIdx<TIndex, origin::Grid, unit::Kernels, dim::D3>::getIdx(index, workExtent));
                return v3uiGridKernelIdx[2]*v3uiGridKernelSize[1]*v3uiGridKernelSize[0] + v3uiGridKernelIdx[1]*v3uiGridKernelSize[0] + v3uiGridKernelIdx[0];
            }
        };
        template<typename TIndex>
        struct GetIdx<TIndex, origin::Grid, unit::Blocks, dim::D3>
        {
            //-----------------------------------------------------------------------------
            //! \return The 3-dimensional index of the current block in the grid.
            //-----------------------------------------------------------------------------
            template<typename TWorkExtent>
            ALPAKA_FCT_HOST_ACC static DimToRetType<dim::D3>::type getIdx(TIndex const & index, IWorkExtent<TWorkExtent> const & )
            {
                return index.getIdxGridBlock();
            }
        };
        template<typename TIndex>
        struct GetIdx<TIndex, origin::Grid, unit::Blocks, dim::Linear>
        {
            //-----------------------------------------------------------------------------
            //! \return The linearized index of the current block in the grid.
            //-----------------------------------------------------------------------------
            template<typename TWorkExtent>
            ALPAKA_FCT_HOST_ACC static DimToRetType<dim::Linear>::type getIdx(TIndex const & index, IWorkExtent<TWorkExtent> const & workExtent)
            {
                auto const v3uiGridBlocksExtent(workExtent.template getExtent<origin::Grid, unit::Blocks, dim::D3>());
                auto const v3uiGridBlockIdx(GetIdx<TIndex, origin::Grid, unit::Blocks, dim::D3>::getIdx(index, workExtent));
                return v3uiGridBlockIdx[2]*v3uiGridBlocksExtent[1]*v3uiGridBlocksExtent[0] + v3uiGridBlockIdx[1]*v3uiGridBlocksExtent[0] + v3uiGridBlockIdx[0];
            }
        };

        //#############################################################################
        //! The index provider interface.
        //#############################################################################
        template<typename TIndex>
        class IIndex :
            private TIndex
        {
        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            template<typename... TArgs>
            ALPAKA_FCT_HOST_ACC IIndex(TArgs && ... args) :
                TIndex(std::forward<TArgs>(args)...)
            {}

            //-----------------------------------------------------------------------------
            //! Get the index requested.
            //-----------------------------------------------------------------------------
            template<typename TOrigin, typename TUnit, typename TDimensionality = dim::D3, typename TWorkExtent = host::detail::WorkExtentHost>
            ALPAKA_FCT_HOST_ACC typename DimToRetType<TDimensionality>::type getIdx(IWorkExtent<TWorkExtent> const & workExtent) const
            {
                return GetIdx<TIndex, TOrigin, TUnit, TDimensionality>::getIdx(
                    *static_cast<TIndex const *>(this),
                    workExtent);
            }
        };
    }
}
