/**
* Copyright 2014 Benjamin Worpitz
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

#include <alpaka/Positioning.hpp>   // alpaka::origin::Grid/Blocks
#include <alpaka/FctCudaCpu.hpp>    // ALPAKA_FCT_CPU_CUDA

#include <utility>                  // std::forward

namespace alpaka
{
    namespace detail
    {
        //#############################################################################
        //! The description of the work size.
        //! This class stores the sizes as members.
        //#############################################################################
        class WorkSizeDefault
        {
        public:
            //-----------------------------------------------------------------------------
            //! Default-constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_CPU_CUDA WorkSizeDefault() = default;
            //-----------------------------------------------------------------------------
            //! Constructor from values.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_CPU_CUDA explicit WorkSizeDefault(vec<3u> const v3uiSizeGridBlocks, vec<3u> const v3uiSizeBlockKernels) :
                m_v3uiSizeGridBlocks(v3uiSizeGridBlocks),
                m_v3uiSizeBlockKernels(v3uiSizeBlockKernels)
            {}
            //-----------------------------------------------------------------------------
            //! Copy-constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_CPU_CUDA WorkSizeDefault(WorkSizeDefault const & other) = default;

            //-----------------------------------------------------------------------------
            //! \return The grid dimensions of the currently executed kernel.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_CPU_CUDA vec<3u> getSizeGridBlocks() const
            {
                return m_v3uiSizeGridBlocks;
            }
            //-----------------------------------------------------------------------------
            //! \return The block dimensions of the currently executed kernel.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_CPU_CUDA vec<3u> getSizeBlockKernels() const
            {
                return m_v3uiSizeBlockKernels;
            }

        private:
            vec<3u> m_v3uiSizeGridBlocks;
            vec<3u> m_v3uiSizeBlockKernels;
        };

        //#############################################################################
        //! The template to get the requested size.
        //#############################################################################
        template<typename TWorkSize, typename TOrigin, typename TUnit, typename TDimensionality>
        struct GetSize;

        template<typename TWorkSize>
        struct GetSize<TWorkSize, origin::Block, unit::Kernels, dim::D3>
        {
            //-----------------------------------------------------------------------------
            //! \return The number of kernels in each dimension of a block.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_CPU_CUDA static DimToRetType<dim::D3>::type getSize(TWorkSize const & workSize)
            {
                return workSize.getSizeBlockKernels();
            }
        };
        template<typename TWorkSize>
        struct GetSize<TWorkSize, origin::Block, unit::Kernels, dim::Linear>
        {
            //-----------------------------------------------------------------------------
            //! \return The number of kernels in a block.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_CPU_CUDA static DimToRetType<dim::Linear>::type getSize(TWorkSize const & workSize)
            {
                return GetSize<TWorkSize, origin::Block, unit::Kernels, dim::D3>::getSize(workSize).prod();
            }
        };
        template<typename TWorkSize>
        struct GetSize<TWorkSize, origin::Grid, unit::Kernels, dim::D3>
        {
            //-----------------------------------------------------------------------------
            //! \return The number of kernels in each dimension of the grid.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_CPU_CUDA static DimToRetType<dim::D3>::type getSize(TWorkSize const & workSize)
            {
                return workSize.getSizeGridBlocks() * workSize.getSizeBlockKernels();
            }
        };
        template<typename TWorkSize>
        struct GetSize<TWorkSize, origin::Grid, unit::Kernels, dim::Linear>
        {
            //-----------------------------------------------------------------------------
            //! \return The number of kernels in the grid.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_CPU_CUDA static DimToRetType<dim::Linear>::type getSize(TWorkSize const & workSize)
            {
                return GetSize<TWorkSize, origin::Grid, unit::Kernels, dim::D3>::getSize(workSize).prod();
            }
        };
        template<typename TWorkSize>
        struct GetSize<TWorkSize, origin::Grid, unit::Blocks, dim::D3>
        {
            //-----------------------------------------------------------------------------
            //! \return The number of blocks in each dimension of the grid.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_CPU_CUDA static DimToRetType<dim::D3>::type getSize(TWorkSize const & workSize)
            {
                return workSize.getSizeGridBlocks();
            }
        };
        template<typename TWorkSize>
        struct GetSize<TWorkSize, origin::Grid, unit::Blocks, dim::Linear>
        {
            //-----------------------------------------------------------------------------
            //! \return The number of blocks in the grid.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_CPU_CUDA static DimToRetType<dim::Linear>::type getSize(TWorkSize const & workSize)
            {
                return GetSize<TWorkSize, origin::Grid, unit::Blocks, dim::D3>::getSize(workSize).prod();
            }
        };
    }

    //#############################################################################
    //! The interface of the work size.
    // TODO: Rename into e.g.: Layout, Subdivision, Partition, Segmentation, Decomposition
    //#############################################################################
    template<typename TWorkSize>
    class IWorkSize :
        private TWorkSize
    {
        //-----------------------------------------------------------------------------
        //! Stream out operator.
        //-----------------------------------------------------------------------------
        template<typename TWorkSize2>
        friend std::ostream & operator << (std::ostream & os, IWorkSize<TWorkSize2> const & workSize);

    public:
        //-----------------------------------------------------------------------------
        //! Constructor.
        //-----------------------------------------------------------------------------
        template<typename... TArgs>
        ALPAKA_FCT_CPU_CUDA IWorkSize(TArgs && ... args) :
            TWorkSize(std::forward<TArgs>(args)...)
        {}

        //-----------------------------------------------------------------------------
        //! Get the size requested.
        //-----------------------------------------------------------------------------
        template<typename TOrigin, typename TUnit, typename TDimensionality = dim::D3>
        ALPAKA_FCT_CPU_CUDA typename detail::DimToRetType<TDimensionality>::type getSize() const
        {
            return detail::GetSize<TWorkSize, TOrigin, TUnit, TDimensionality>::getSize(
                *static_cast<TWorkSize const *>(this));
        }
    };

    //-----------------------------------------------------------------------------
    //! Stream out operator.
    //-----------------------------------------------------------------------------
    template<typename TWorkSize>
    ALPAKA_FCT_CPU std::ostream & operator << (std::ostream & os, IWorkSize<TWorkSize> const & workSize)
    {
        return (os << "{GridBlocks: " << workSize.getSizeGridBlocks() << ", BlockKernels: " << workSize.getSizeBlockKernels() << "}");
    }

    //#############################################################################
    //! A basic class storing the work to be used in user code.
    //#############################################################################
    using WorkSize = IWorkSize<detail::WorkSizeDefault>;
}