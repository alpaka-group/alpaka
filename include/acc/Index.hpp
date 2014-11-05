/**
* Copyright 2014 Benjamin Worpitz
*
* This file is part of acc.
*
* acc is free software: you can redistribute it and/or modify
* it under the terms of of either the GNU General Public License or
* the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* acc is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License and the GNU Lesser General Public License
* for more details.
*
* You should have received a copy of the GNU General Public License
* and the GNU Lesser General Public License along with acc.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <acc/Positioning.hpp>    // acc::origin::Grid/Tiles
#include <acc/WorkSize.hpp>        // acc::IWorkSize
#include <acc/FctCudaCpu.hpp>    // ACC_FCT_CPU_CUDA

#include <utility>                // std::forward

namespace acc
{
    namespace detail
    {
        //#############################################################################
        //! The template declaration to get the requsted index.
        //#############################################################################
        template<typename TWorkSize, typename TIndex, typename TOrigin, typename TUnit, typename TDimensionality>
        struct GetIdx;

        //#############################################################################
        //! The extended description of the work size.
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
            ACC_FCT_CPU_CUDA IIndex(TArgs && ... args) :
                TIndex(std::forward<TArgs>(args)...)
            {}

            //-----------------------------------------------------------------------------
            //! Get the index requested.
            // Member function templates of a template class can not be specialized.
            // Therefore this is done via the partial specialized getIdx class, which gets a reference to this object.
            //-----------------------------------------------------------------------------
            template<typename TWorkSize, typename TOrigin, typename TUnit, typename TDimensionality = dim::D3>
            ACC_FCT_CPU_CUDA typename DimToRetType<TDimensionality>::type getIdx(TWorkSize const & workSize) const
            {
                return GetIdx<TWorkSize, TIndex, TOrigin, TUnit, TDimensionality>::getIdx(
                    *static_cast<TIndex const *>(this),
                    workSize);
            }
        };

        //#############################################################################
        //! The template specializations to get the requsted index.
        //#############################################################################
        template<typename TWorkSize, typename TIndex>
        struct GetIdx<TWorkSize, TIndex, origin::Tile, unit::Kernels, dim::D3>
        {
            //-----------------------------------------------------------------------------
            //! \return The 3-dimensional index of the current kernel in the tile.
            //-----------------------------------------------------------------------------
            ACC_FCT_CPU_CUDA static DimToRetType<dim::D3>::type getIdx(TIndex const & index, TWorkSize const & )
            {
                return index.getIdxTileKernel();
            }
        };
        template<typename TWorkSize, typename TIndex>
        struct GetIdx<TWorkSize, TIndex, origin::Tile, unit::Kernels, dim::Linear>
        {
            //-----------------------------------------------------------------------------
            //! \return The linearized index of the current kernel in the tile.
            //-----------------------------------------------------------------------------
            ACC_FCT_CPU_CUDA static DimToRetType<dim::Linear>::type getIdx(TIndex const & index, TWorkSize const & workSize)
            {
                auto const v3uiSizeTileKernels(GetSize<TWorkSize, origin::Tile, unit::Kernels, dim::D3>()::getSize(workSize));
                auto const v3uiTileKernelIdx(GetIdx<TIndex, TWorkSize, origin::Tile, unit::Kernels, dim::D3>::getIdx(index, workSize));
                return v3uiTileKernelIdx[2]*v3uiSizeTileKernels[1]*v3uiSizeTileKernels[0] + v3uiTileKernelIdx[1]*v3uiSizeTileKernels[0] + v3uiTileKernelIdx[0];
            }
        };
        template<typename TWorkSize, typename TIndex>
        struct GetIdx<TWorkSize, TIndex, origin::Grid, unit::Kernels, dim::D3>
        {
            //-----------------------------------------------------------------------------
            //! \return The 3-dimensional index of the current kernel in grid.
            //-----------------------------------------------------------------------------
            ACC_FCT_CPU_CUDA static DimToRetType<dim::D3>::type getIdx(TIndex const & index, TWorkSize const & workSize)
            {
                return 
                    index.getIdxGridTile() * GetSize<TIndex, origin::Tile, unit::Kernels, dim::D3>()::getSize(workSize) 
                    + index.getIdxTileKernel();
            }
        };
        template<typename TWorkSize, typename TIndex>
        struct GetIdx<TWorkSize, TIndex, origin::Grid, unit::Kernels, dim::Linear>
        {
            //-----------------------------------------------------------------------------
            //! \return The linearized index of the current kernel in the grid.
            //-----------------------------------------------------------------------------
            ACC_FCT_CPU_CUDA static DimToRetType<dim::Linear>::type getIdx(TIndex const & index, TWorkSize const & workSize)
            {
                auto const v3uiGridKernelSize(GetSize<TWorkSize, origin::Grid, unit::Kernels, dim::D3>()::getSize(workSize));
                auto const v3uiGridKernelIdx(GetIdx<TIndex, TWorkSize, origin::Grid, unit::Kernels, dim::D3>::getIdx(index, workSize));
                return v3uiGridKernelIdx[2]*v3uiGridKernelSize[1]*v3uiGridKernelSize[0] + v3uiGridKernelIdx[1]*v3uiGridKernelSize[0] + v3uiGridKernelIdx[0];
            }
        };
        template<typename TWorkSize, typename TIndex>
        struct GetIdx<TWorkSize, TIndex, origin::Grid, unit::Tiles, dim::D3>
        {
            //-----------------------------------------------------------------------------
            //! \return The 3-dimensional index of the current tile in the grid.
            //-----------------------------------------------------------------------------
            ACC_FCT_CPU_CUDA static DimToRetType<dim::D3>::type getIdx(TIndex const & index, TWorkSize const & )
            {
                return index.getIdxGridTile();
            }
        };
        template<typename TWorkSize, typename TIndex>
        struct GetIdx<TWorkSize, TIndex, origin::Grid, unit::Tiles, dim::Linear>
        {
            //-----------------------------------------------------------------------------
            //! \return The linearized index of the current tile in the grid.
            //-----------------------------------------------------------------------------
            ACC_FCT_CPU_CUDA static DimToRetType<dim::Linear>::type getIdx(TIndex const & index, TWorkSize const & workSize)
            {
                auto const v3uiSizeGridTiles(GetSize<TWorkSize, origin::Grid, unit::Tiles, dim::D3>()::getSize(workSize));
                auto const v3uiGridTileIdx(GetIdx<TIndex, TWorkSize, origin::Grid, unit::Tiles, dim::D3>::getIdx(index, workSize));
                return v3uiGridTileIdx[2]*v3uiSizeGridTiles[1]*v3uiSizeGridTiles[0] + v3uiGridTileIdx[1]*v3uiSizeGridTiles[0] + v3uiGridTileIdx[0];
            }
        };
    }
}