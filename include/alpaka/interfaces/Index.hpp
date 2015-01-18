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

#include <alpaka/traits/Dim.hpp>            // alpaka::dim::DimToVecT

#include <alpaka/core/BasicDims.hpp>        // alpaka::dim::Dim<N>
#include <alpaka/core/Positioning.hpp>      // alpaka::origin::Grid/Blocks, alpaka::unit::Blocks, alpaka::unit::Kernels
#include <alpaka/core/Common.hpp>           // ALPAKA_FCT_ACC

#include <alpaka/host/WorkExtent.hpp>       // alpaka::WorkExtentHost

#include <utility>                          // std::forward

namespace alpaka
{
    namespace traits
    {
        //#############################################################################
        //! The index get trait.
        //#############################################################################
        template<
            typename TIndex, 
            typename TOrigin, 
            typename TUnit, 
            typename TDimensionality>
        struct GetIdx;

        //#############################################################################
        //! The 3D block kernels index get trait specialization.
        //#############################################################################
        template<
            typename TIndex>
        struct GetIdx<
            TIndex, 
            origin::Block, 
            unit::Kernels, 
            alpaka::dim::Dim3>
        {
            //-----------------------------------------------------------------------------
            //! \return The 3-dimensional index of the current kernel in the block.
            //-----------------------------------------------------------------------------
            template<
                typename TWorkExtent>
            ALPAKA_FCT_ACC static alpaka::dim::DimToVecT<alpaka::dim::Dim3> getIdx(
                TIndex const & index, 
                IWorkExtent<TWorkExtent> const &)
            {
                return index.getIdxBlockKernel();
            }
        };
        //#############################################################################
        //! The 1D block kernels index get trait specialization.
        //#############################################################################
        template<
            typename TIndex>
        struct GetIdx<
            TIndex, 
            origin::Block, 
            unit::Kernels, 
            alpaka::dim::Dim1>
        {
            //-----------------------------------------------------------------------------
            //! \return The linearized index of the current kernel in the block.
            //-----------------------------------------------------------------------------
            template<
                typename TWorkExtent>
            ALPAKA_FCT_ACC static alpaka::dim::DimToVecT<alpaka::dim::Dim1> getIdx(
                TIndex const & index, 
                IWorkExtent<TWorkExtent> const & workExtent)
            {
                auto const v3uiBlockKernelsExtent(workExtent.template getExtent<origin::Block, unit::Kernels, alpaka::dim::Dim3>());
                auto const v3uiBlockKernelIdx(GetIdx<TIndex, origin::Block, unit::Kernels, alpaka::dim::Dim3>::getIdx(index, workExtent));
                return v3uiBlockKernelIdx[2] * v3uiBlockKernelsExtent[1] * v3uiBlockKernelsExtent[0] + v3uiBlockKernelIdx[1] * v3uiBlockKernelsExtent[0] + v3uiBlockKernelIdx[0];
            }
        };
        //#############################################################################
        //! The 3D grid kernels index get trait specialization.
        //#############################################################################
        template<
            typename TIndex>
        struct GetIdx<
            TIndex, 
            origin::Grid, 
            unit::Kernels, 
            alpaka::dim::Dim3>
        {
            //-----------------------------------------------------------------------------
            //! \return The 3-dimensional index of the current kernel in grid.
            //-----------------------------------------------------------------------------
            template<
                typename TWorkExtent>
            ALPAKA_FCT_ACC static alpaka::dim::DimToVecT<alpaka::dim::Dim3> getIdx(
                TIndex const & index, 
                IWorkExtent<TWorkExtent> const & workExtent)
            {
                return
                    index.getIdxGridBlock() * workExtent.template getExtent<origin::Block, unit::Kernels, alpaka::dim::Dim3>()
                    + index.getIdxBlockKernel();
            }
        };
        //#############################################################################
        //! The 1D grid kernels index get trait specialization.
        //#############################################################################
        template<
            typename TIndex>
        struct GetIdx<
            TIndex, 
            origin::Grid, 
            unit::Kernels, 
            alpaka::dim::Dim1>
        {
            //-----------------------------------------------------------------------------
            //! \return The linearized index of the current kernel in the grid.
            //-----------------------------------------------------------------------------
            template<
                typename TWorkExtent>
            ALPAKA_FCT_ACC static alpaka::dim::DimToVecT<alpaka::dim::Dim1> getIdx(
                TIndex const & index, 
                IWorkExtent<TWorkExtent> const & workExtent)
            {
                auto const v3uiGridKernelSize(workExtent.template getExtent<origin::Grid, unit::Kernels, alpaka::dim::Dim3>());
                auto const v3uiGridKernelIdx(GetIdx<TIndex, origin::Grid, unit::Kernels, alpaka::dim::Dim3>::getIdx(index, workExtent));
                return v3uiGridKernelIdx[2] * v3uiGridKernelSize[1] * v3uiGridKernelSize[0] + v3uiGridKernelIdx[1] * v3uiGridKernelSize[0] + v3uiGridKernelIdx[0];
            }
        };
        //#############################################################################
        //! The 3D grid blocks index get trait specialization.
        //#############################################################################
        template<
            typename TIndex>
        struct GetIdx<
            TIndex, 
            origin::Grid, 
            unit::Blocks, 
            alpaka::dim::Dim3>
        {
            //-----------------------------------------------------------------------------
            //! \return The 3-dimensional index of the current block in the grid.
            //-----------------------------------------------------------------------------
            template<
                typename TWorkExtent>
            ALPAKA_FCT_ACC static alpaka::dim::DimToVecT<alpaka::dim::Dim3> getIdx(
                TIndex const & index, 
                IWorkExtent<TWorkExtent> const &)
            {
                return index.getIdxGridBlock();
            }
        };
        //#############################################################################
        //! The 1D grid blocks index get trait specialization.
        //#############################################################################
        template<
            typename TIndex>
        struct GetIdx<
            TIndex, 
            origin::Grid, 
            unit::Blocks, 
            alpaka::dim::Dim1>
        {
            //-----------------------------------------------------------------------------
            //! \return The linearized index of the current block in the grid.
            //-----------------------------------------------------------------------------
            template<
                typename TWorkExtent>
            ALPAKA_FCT_ACC static alpaka::dim::DimToVecT<alpaka::dim::Dim1> getIdx(
                TIndex const & index, 
                IWorkExtent<TWorkExtent> const & workExtent)
            {
                auto const v3uiGridBlocksExtent(workExtent.template getExtent<origin::Grid, unit::Blocks, alpaka::dim::Dim3>());
                auto const v3uiGridBlockIdx(GetIdx<TIndex, origin::Grid, unit::Blocks, alpaka::dim::Dim3>::getIdx(index, workExtent));
                return v3uiGridBlockIdx[2] * v3uiGridBlocksExtent[1] * v3uiGridBlocksExtent[0] + v3uiGridBlockIdx[1] * v3uiGridBlocksExtent[0] + v3uiGridBlockIdx[0];
            }
        };
    }

    namespace detail
    {
        //#############################################################################
        //! The index provider interface.
        //#############################################################################
        template<
            typename TIndex>
        class IIndex :
            private TIndex
        {
        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            template<
                typename... TArgs>
            ALPAKA_FCT_ACC IIndex(
                TArgs && ... args) :
                TIndex(std::forward<TArgs>(args)...)
            {}

            //-----------------------------------------------------------------------------
            //! Get the index requested.
            //-----------------------------------------------------------------------------
            template<
                typename TOrigin, 
                typename TUnit, 
                typename TDimensionality = dim::Dim3, 
                typename TWorkExtent = host::detail::WorkExtentHost>
            ALPAKA_FCT_ACC typename dim::DimToVecT<TDimensionality> getIdx(
                IWorkExtent<TWorkExtent> const & workExtent) const
            {
                return traits::GetIdx<TIndex, TOrigin, TUnit, TDimensionality>::getIdx(
                    *static_cast<TIndex const *>(this),
                    workExtent);
            }
        };
    }
}
