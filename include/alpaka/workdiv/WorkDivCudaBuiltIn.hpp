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

#include <alpaka/workdiv/Traits.hpp>                // workdiv::GetWorkDiv

#include <alpaka/core/Vec.hpp>                      // Vec, getExtentsVecNd
#include <alpaka/core/Common.hpp>                   // ALPAKA_FCT_ACC_CUDA_ONLY
#include <alpaka/core/Cuda.hpp>                     // getExtent(dim3)

//#include <boost/core/ignore_unused.hpp>           // boost::ignore_unused

namespace alpaka
{
    namespace workdiv
    {
        //#############################################################################
        //! The GPU CUDA accelerator work division.
        //#############################################################################
        template<
            typename TDim>
        class WorkDivCudaBuiltIn
        {
        public:
            using WorkDivBase = WorkDivCudaBuiltIn;

            //-----------------------------------------------------------------------------
            //! Default constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_ACC_CUDA_ONLY WorkDivCudaBuiltIn() = default;
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_ACC_CUDA_ONLY WorkDivCudaBuiltIn(WorkDivCudaBuiltIn const &) = delete;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_ACC_CUDA_ONLY WorkDivCudaBuiltIn(WorkDivCudaBuiltIn &&) = delete;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_ACC_CUDA_ONLY auto operator=(WorkDivCudaBuiltIn const &) -> WorkDivCudaBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_ACC_CUDA_ONLY auto operator=(WorkDivCudaBuiltIn &&) -> WorkDivCudaBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_ACC_CUDA_ONLY /*virtual*/ ~WorkDivCudaBuiltIn() = default;
        };
    }

    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator work division dimension get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DimType<
                workdiv::WorkDivCudaBuiltIn<TDim>>
            {
                using type = TDim;
            };
        }
    }
    namespace workdiv
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator work division block thread extents trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetWorkDiv<
                WorkDivCudaBuiltIn<TDim>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of threads in each dimension of a block.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY static auto getWorkDiv(
                    WorkDivCudaBuiltIn<TDim> const & workDiv)
                -> alpaka::Vec<TDim>
                {
                    //boost::ignore_unused(workDiv);
                    return extent::getExtentsVecNd<TDim, UInt>(blockDim);
                }
            };

            //#############################################################################
            //! The GPU CUDA accelerator work division grid block extents trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetWorkDiv<
                WorkDivCudaBuiltIn<TDim>,
                origin::Grid,
                unit::Blocks>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of blocks in each dimension of the grid.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY static auto getWorkDiv(
                    WorkDivCudaBuiltIn<TDim> const & workDiv)
                -> alpaka::Vec<TDim>
                {
                    //boost::ignore_unused(workDiv);
                    return extent::getExtentsVecNd<TDim, UInt>(gridDim);
                }
            };
        }
    }
}
