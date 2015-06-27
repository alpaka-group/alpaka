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

#include <alpaka/idx/Traits.hpp>                // idx::getIdx

#include <alpaka/core/Vec.hpp>                  // Vec, getOffsetsVecNd
#include <alpaka/core/Cuda.hpp>                 // getOffset(dim3)

//#include <boost/core/ignore_unused.hpp>         // boost::ignore_unused

namespace alpaka
{
    namespace idx
    {
        namespace gb
        {
            //#############################################################################
            //! The CUDA accelerator ND index provider.
            //#############################################################################
            template<
                typename TDim>
            class IdxGbCudaBuiltIn
            {
            public:
                using IdxGbBase = IdxGbCudaBuiltIn;

                //-----------------------------------------------------------------------------
                //! Default constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY IdxGbCudaBuiltIn() = default;
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY IdxGbCudaBuiltIn(IdxGbCudaBuiltIn const &) = delete;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY IdxGbCudaBuiltIn(IdxGbCudaBuiltIn &&) = delete;
                //-----------------------------------------------------------------------------
                //! Copy assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY auto operator=(IdxGbCudaBuiltIn const & ) -> IdxGbCudaBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                //! Move assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY auto operator=(IdxGbCudaBuiltIn &&) -> IdxGbCudaBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY /*virtual*/ ~IdxGbCudaBuiltIn() = default;
            };
        }
    }

    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator index dimension get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DimType<
                idx::gb::IdxGbCudaBuiltIn<TDim>>
            {
                using type = TDim;
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator grid block index get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetIdx<
                gb::IdxGbCudaBuiltIn<TDim>,
                origin::Grid,
                unit::Blocks>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current block in the grid.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_ACC_CUDA_ONLY static auto getIdx(
                    gb::IdxGbCudaBuiltIn<TDim> const & idx,
                    TWorkDiv const &)
                -> Vec<TDim>
                {
                    //boost::ignore_unused(idx);
                    return offset::getOffsetsVecNd<TDim, Uint>(blockIdx);
                }
            };
        }
    }
}
