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
        namespace bt
        {
            //#############################################################################
            //! The CUDA accelerator ND index provider.
            //#############################################################################
            template<
                typename TDim>
            class IdxBtCudaBuiltIn
            {
            public:
                using IdxBtBase = IdxBtCudaBuiltIn;

                //-----------------------------------------------------------------------------
                //! Default constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY IdxBtCudaBuiltIn() = default;
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY IdxBtCudaBuiltIn(IdxBtCudaBuiltIn const &) = delete;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY IdxBtCudaBuiltIn(IdxBtCudaBuiltIn &&) = delete;
                //-----------------------------------------------------------------------------
                //! Copy assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY auto operator=(IdxBtCudaBuiltIn const & ) -> IdxBtCudaBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                //! Move assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY auto operator=(IdxBtCudaBuiltIn &&) -> IdxBtCudaBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY /*virtual*/ ~IdxBtCudaBuiltIn() = default;
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
                idx::bt::IdxBtCudaBuiltIn<TDim>>
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
            //! The GPU CUDA accelerator block thread index get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetIdx<
                bt::IdxBtCudaBuiltIn<TDim>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current thread in the block.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_ACC_CUDA_ONLY static auto getIdx(
                    bt::IdxBtCudaBuiltIn<TDim> const & idx,
                    TWorkDiv const &)
                -> Vec<TDim>
                {
                    //boost::ignore_unused(idx);
                    return offset::getOffsetsVecNd<TDim, Uint>(threadIdx);
                }
            };
        }
    }
}
