/**
* Copyright 2014 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of of either the GNU General Public License or
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

#include <alpaka/Positioning.hpp>   // alpaka::origin::Grid/Blocks
#include <alpaka/WorkSize.hpp>      // alpaka::IWorkSize
#include <alpaka/FctCudaCpu.hpp>    // ALPAKA_FCT_CPU_CUDA

#include <utility>                  // std::forward

namespace alpaka
{
    namespace detail
    {
        //#############################################################################
        //! The template declaration to get the requsted index.
        //#############################################################################
        template<typename TPackedWorkSize, typename TIndex, typename TOrigin, typename TUnit, typename TDimensionality>
        struct GetIdx;

        //#############################################################################
        //! The extended description of the work size.
        //#############################################################################
        template<typename TIndex>
        class IIndex :
            private TIndex
        {
        public:
            using TIndexImpl = TIndex;

        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            template<typename... TArgs>
            ALPAKA_FCT_CPU_CUDA IIndex(TArgs && ... args) :
                TIndex(std::forward<TArgs>(args)...)
            {}

            //-----------------------------------------------------------------------------
            //! Get the index requested.
            // Member function templates of a template class can not be specialized.
            // Therefore this is done via the partial specialized getIdx class, which gets a reference to this object.
            //-----------------------------------------------------------------------------
            template<typename TPackedWorkSize, typename TOrigin, typename TUnit, typename TDimensionality = dim::D3>
            ALPAKA_FCT_CPU_CUDA typename DimToRetType<TDimensionality>::type getIdx(TPackedWorkSize const & workSize) const
            {
                return GetIdx<typename TPackedWorkSize, TIndex, TOrigin, TUnit, TDimensionality>::getIdx(
                    *static_cast<TIndex const *>(this),
                    workSize);
            }
        };

        //#############################################################################
        //! The template specializations to get the requsted index.
        //#############################################################################
        template<typename TPackedWorkSize, typename TIndex>
        struct GetIdx<TPackedWorkSize, TIndex, origin::Block, unit::Kernels, dim::D3>
        {
            //-----------------------------------------------------------------------------
            //! \return The 3-dimensional index of the current kernel in the block.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_CPU_CUDA static DimToRetType<dim::D3>::type getIdx(TIndex const & index, TPackedWorkSize const &)
            {
                return index.getIdxBlockKernel();
            }
        };
        template<typename TPackedWorkSize, typename TIndex>
        struct GetIdx<TPackedWorkSize, TIndex, origin::Block, unit::Kernels, dim::Linear>
        {
            //-----------------------------------------------------------------------------
            //! \return The linearized index of the current kernel in the block.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_CPU_CUDA static DimToRetType<dim::Linear>::type getIdx(TIndex const & index, TPackedWorkSize const & workSize)
            {
                auto const v3uiSizeBlockKernels(workSize.template getSize<origin::Block, unit::Kernels, dim::D3>());
                auto const v3uiBlockKernelIdx(GetIdx<TPackedWorkSize, TIndex, origin::Block, unit::Kernels, dim::D3>::getIdx(index, workSize));
                return v3uiBlockKernelIdx[2]*v3uiSizeBlockKernels[1]*v3uiSizeBlockKernels[0] + v3uiBlockKernelIdx[1]*v3uiSizeBlockKernels[0] + v3uiBlockKernelIdx[0];
            }
        };
        template<typename TPackedWorkSize, typename TIndex>
        struct GetIdx<TPackedWorkSize, TIndex, origin::Grid, unit::Kernels, dim::D3>
        {
            //-----------------------------------------------------------------------------
            //! \return The 3-dimensional index of the current kernel in grid.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_CPU_CUDA static DimToRetType<dim::D3>::type getIdx(TIndex const & index, TPackedWorkSize const & workSize)
            {
                return 
                    index.getIdxGridBlock() * GetSize<TIndex, origin::Block, unit::Kernels, dim::D3>()::getSize(workSize) 
                    + index.getIdxBlockKernel();
            }
        };
        template<typename TPackedWorkSize, typename TIndex>
        struct GetIdx<TPackedWorkSize, TIndex, origin::Grid, unit::Kernels, dim::Linear>
        {
            //-----------------------------------------------------------------------------
            //! \return The linearized index of the current kernel in the grid.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_CPU_CUDA static DimToRetType<dim::Linear>::type getIdx(TIndex const & index, TPackedWorkSize const & workSize)
            {
                auto const v3uiGridKernelSize(workSize.template getSize<origin::Grid, unit::Kernels, dim::D3>());
                auto const v3uiGridKernelIdx(GetIdx<TPackedWorkSize, TIndex, origin::Grid, unit::Kernels, dim::D3>::getIdx(index, workSize));
                return v3uiGridKernelIdx[2]*v3uiGridKernelSize[1]*v3uiGridKernelSize[0] + v3uiGridKernelIdx[1]*v3uiGridKernelSize[0] + v3uiGridKernelIdx[0];
            }
        };
        template<typename TPackedWorkSize, typename TIndex>
        struct GetIdx<TPackedWorkSize, TIndex, origin::Grid, unit::Blocks, dim::D3>
        {
            //-----------------------------------------------------------------------------
            //! \return The 3-dimensional index of the current block in the grid.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_CPU_CUDA static DimToRetType<dim::D3>::type getIdx(TIndex const & index, TPackedWorkSize const & )
            {
                return index.getIdxGridBlock();
            }
        };
        template<typename TPackedWorkSize, typename TIndex>
        struct GetIdx<TPackedWorkSize, TIndex, origin::Grid, unit::Blocks, dim::Linear>
        {
            //-----------------------------------------------------------------------------
            //! \return The linearized index of the current block in the grid.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_CPU_CUDA static DimToRetType<dim::Linear>::type getIdx(TIndex const & index, TPackedWorkSize const & workSize)
            {
                auto const v3uiSizeGridBlocks(workSize.template getSize<origin::Grid, unit::Blocks, dim::D3>());
                auto const v3uiGridBlockIdx(GetIdx<TPackedWorkSize, TIndex, origin::Grid, unit::Blocks, dim::D3>::getIdx(index, workSize));
                return v3uiGridBlockIdx[2]*v3uiSizeGridBlocks[1]*v3uiSizeGridBlocks[0] + v3uiGridBlockIdx[1]*v3uiSizeGridBlocks[0] + v3uiGridBlockIdx[0];
            }
        };
    }
}