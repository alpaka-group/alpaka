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

#include <alpaka/traits/WorkDiv.hpp>    // alpaka::GetWorkDiv

#include <alpaka/core/Vec.hpp>          // Vec, getExtentsVecNd
#include <alpaka/core/Common.hpp>       // ALPAKA_FCT_ACC_CUDA_ONLY
#include <alpaka/core/Cuda.hpp>  // gridDim, blockDim, getExtent(dim3)

namespace alpaka
{
    namespace accs
    {
        namespace cuda
        {
            namespace detail
            {
                //#############################################################################
                //! The GPU CUDA accelerator work division.
                //#############################################################################
                template<
                    typename TDim>
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
                    ALPAKA_FCT_ACC_CUDA_ONLY WorkDivCuda(WorkDivCuda const &) = delete;
    #if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_CUDA_ONLY WorkDivCuda(WorkDivCuda &&) = delete;
    #endif
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_CUDA_ONLY auto operator=(WorkDivCuda const &) -> WorkDivCuda & = delete;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_CUDA_ONLY auto operator=(WorkDivCuda &&) -> WorkDivCuda & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_CUDA_ONLY /*virtual*/ ~WorkDivCuda() noexcept = default;

                    //-----------------------------------------------------------------------------
                    //! \return The grid block extents of the currently executed thread.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_CUDA_ONLY auto getGridBlockExtents() const
                    -> Vec<TDim>
                    {
                        return extent::getExtentsVecNd<TDim, UInt>(gridDim);
                    }
                    //-----------------------------------------------------------------------------
                    //! \return The block thread extents of the currently executed thread.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_CUDA_ONLY auto getBlockThreadExtents() const
                    -> Vec<TDim>
                    {
                        return extent::getExtentsVecNd<TDim, UInt>(blockDim);
                    }
                };
            }
        }
    }

    namespace traits
    {
        namespace dim
        {
            //#############################################################################
            //! The GPU CUDA accelerator work division dimension get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DimType<
                accs::cuda::detail::WorkDivCuda<TDim>>
            {
                using type = TDim;
            };
        }

        namespace workdiv
        {
            //#############################################################################
            //! The GPU CUDA accelerator work division block thread 3D extents trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetWorkDiv<
                accs::cuda::detail::WorkDivCuda<TDim>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of threads in each dimension of a block.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY static auto getWorkDiv(
                    accs::cuda::detail::WorkDivCuda<TDim> const & workDiv)
                -> alpaka::Vec<TDim>
                {
                    return workDiv.getBlockThreadExtents();
                }
            };

            //#############################################################################
            //! The GPU CUDA accelerator work division grid block 3D extents trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetWorkDiv<
                accs::cuda::detail::WorkDivCuda<TDim>,
                origin::Grid,
                unit::Blocks>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of blocks in each dimension of the grid.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY static auto getWorkDiv(
                    accs::cuda::detail::WorkDivCuda<TDim> const & workDiv)
                -> alpaka::Vec<TDim>
                {
                    return workDiv.getGridBlockExtents();
                }
            };
        }
    }
}
