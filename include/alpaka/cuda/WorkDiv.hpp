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

#include <alpaka/core/Vec.hpp>          // Vec

#include <alpaka/traits/WorkDiv.hpp>    // alpaka::GetWorkDiv

namespace alpaka
{
    namespace cuda
    {
        namespace detail
        {
            //#############################################################################
            //! The CUDA accelerator work division.
            //#############################################################################
            class WorkDivCuda
            {
            public:
                //-----------------------------------------------------------------------------
                //! Default constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY WorkDivCuda() = default;
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY WorkDivCuda(WorkDivCuda const &) = default;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY WorkDivCuda(WorkDivCuda &&) = default;
                //-----------------------------------------------------------------------------
                //! Copy assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY WorkDivCuda & operator=(WorkDivCuda const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY /*virtual*/ ~WorkDivCuda() noexcept = default;

                //-----------------------------------------------------------------------------
                //! \return The grid blocks extents of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY Vec<3u> getGridBlocksExtents() const
                {
                    return {gridDim.x, gridDim.y, gridDim.z};
                }
                //-----------------------------------------------------------------------------
                //! \return The block kernels extents of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY Vec<3u> getBlockKernelsExtents() const
                {
                    return {blockDim.x, blockDim.y, blockDim.z};
                }
            };
        }
    }

    namespace traits
    {
        namespace workdiv
        {
            //#############################################################################
            //! The CUDA accelerator work div block kernels 3D extents trait specialization.
            //#############################################################################
            template<>
            struct GetWorkDiv<
                cuda::detail::WorkDivCuda,
                origin::Block,
                unit::Kernels,
                alpaka::dim::Dim3>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of kernels in each dimension of a block.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY static alpaka::dim::DimToVecT<alpaka::dim::Dim3> getWorkDiv(
                    cuda::detail::WorkDivCuda const & workDiv)
                {
                    return workDiv.getBlockKernelsExtents();
                }
            };

            //#############################################################################
            //! The CUDA accelerator work div grid blocks 3D extents trait specialization.
            //#############################################################################
            template<>
            struct GetWorkDiv<
                cuda::detail::WorkDivCuda,
                origin::Grid,
                unit::Blocks,
                alpaka::dim::Dim3>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of blocks in each dimension of the grid.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY static alpaka::dim::DimToVecT<alpaka::dim::Dim3> getWorkDiv(
                    cuda::detail::WorkDivCuda const & workDiv)
                {
                    return workDiv.getGridBlocksExtents();
                }
            };
        }
    }
}
