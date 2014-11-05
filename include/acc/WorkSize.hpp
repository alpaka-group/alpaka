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

#include <acc/Positioning.hpp>  // acc::origin::Grid/Blocks
#include <acc/FctCudaCpu.hpp>   // ACC_FCT_CPU_CUDA

#include <utility>              // std::forward

namespace acc
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
            ACC_FCT_CPU_CUDA WorkSizeDefault() = default;

            //-----------------------------------------------------------------------------
            //! Constructor from values.
            //-----------------------------------------------------------------------------
            ACC_FCT_CPU_CUDA explicit WorkSizeDefault(vec<3> const v3uiSizeGridBlocks, vec<3> const v3uiSizeBlockKernels) :
                m_v3uiSizeGridBlocks(v3uiSizeGridBlocks),
                m_v3uiSizeBlockKernels(v3uiSizeBlockKernels)
            {}

            //-----------------------------------------------------------------------------
            //! Copy-onstructor.
            //-----------------------------------------------------------------------------
            ACC_FCT_CPU_CUDA WorkSizeDefault(WorkSizeDefault const & other) = default;

            //-----------------------------------------------------------------------------
            //! \return The grid dimensions of the currently executed kernel.
            //-----------------------------------------------------------------------------
            ACC_FCT_CPU_CUDA vec<3> getSizeGridBlocks() const
            {
                return m_v3uiSizeGridBlocks;
            }
            //-----------------------------------------------------------------------------
            //! \return The block dimensions of the currently executed kernel.
            //-----------------------------------------------------------------------------
            ACC_FCT_CPU_CUDA vec<3> getSizeBlockKernels() const
            {
                return m_v3uiSizeBlockKernels;
            }

        private:
            vec<3> m_v3uiSizeGridBlocks;
            vec<3> m_v3uiSizeBlockKernels;
        };

        //#############################################################################
        //! The template declaration to get the requsted size.
        //#############################################################################
        template<typename TWorkSize, typename TOrigin, typename TUnit, typename TDimensionality>
        struct GetSize;

        //#############################################################################
        //! The extended description of the work size.
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
            using TWorkSizeImpl = TWorkSize;

        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            template<typename... TArgs>
            ACC_FCT_CPU_CUDA IWorkSize(TArgs && ... args) :
                TWorkSize(std::forward<TArgs>(args)...)
            {}

            //-----------------------------------------------------------------------------
            //! Get the size requested.
            // Member function templates of a template class can not be specialized.
            // Therefore this is done via the partial specialized GetSize class, which gets a reference to this object.
            //-----------------------------------------------------------------------------
            template<typename TOrigin, typename TUnit, typename TDimensionality = dim::D3>
            ACC_FCT_CPU_CUDA typename DimToRetType<TDimensionality>::type getSize() const
            {
                return GetSize<TWorkSize, TOrigin, TUnit, TDimensionality>::getSize(
                    *static_cast<TWorkSize const *>(this));
            }
        };

        //#############################################################################
        //! The template specializations to get the requsted size.
        //#############################################################################
        template<typename TWorkSize>
        struct GetSize<TWorkSize, origin::Block, unit::Kernels, dim::D3>
        {
            //-----------------------------------------------------------------------------
            //! \return The number of kernels in each dimension of a block.
            //-----------------------------------------------------------------------------
            ACC_FCT_CPU_CUDA static DimToRetType<dim::D3>::type getSize(TWorkSize const & workSize)
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
            ACC_FCT_CPU_CUDA static DimToRetType<dim::Linear>::type getSize(TWorkSize const & workSize)
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
            ACC_FCT_CPU_CUDA static DimToRetType<dim::D3>::type getSize(TWorkSize const & workSize)
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
            ACC_FCT_CPU_CUDA static DimToRetType<dim::Linear>::type getSize(TWorkSize const & workSize)
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
            ACC_FCT_CPU_CUDA static DimToRetType<dim::D3>::type getSize(TWorkSize const & workSize)
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
            ACC_FCT_CPU_CUDA static DimToRetType<dim::Linear>::type getSize(TWorkSize const & workSize)
            {
                return GetSize<TWorkSize, origin::Grid, unit::Blocks, dim::D3>::getSize(workSize).prod();
            }
        };

        //-----------------------------------------------------------------------------
        //! Stream out operator.
        //-----------------------------------------------------------------------------
        template<typename TWorkSizeImpl>
        ACC_FCT_CPU std::ostream & operator << (std::ostream & os, IWorkSize<TWorkSizeImpl> const & workSize)
        {
            return (os << "{GridBlocks: " << workSize.getSizeGridBlocks() << ", BlockKernels: " << workSize.getSizeBlockKernels() << "}");
        }
    }

    //#############################################################################
    //! A basic class storing the work to be used in user code.
    //#############################################################################
    using WorkSize = detail::IWorkSize<detail::WorkSizeDefault>;
}