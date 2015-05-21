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

#include <alpaka/accs/cuda/Common.hpp>  // threadIdx, blockIdx, getOffset(dim3)

#include <alpaka/traits/Idx.hpp>        // idx::getIdx

#include <alpaka/core/Vec.hpp>          // Vec, getOffsetsVecNd

namespace alpaka
{
    namespace accs
    {
        namespace cuda
        {
            namespace detail
            {
                //#############################################################################
                //! This CUDA accelerator ND index provider.
                //#############################################################################
                template<
                    typename TDim>
                class IdxCuda
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Default constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_CUDA_ONLY IdxCuda() = default;
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_CUDA_ONLY IdxCuda(IdxCuda const &) = default;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_CUDA_ONLY IdxCuda(IdxCuda &&) = default;
#endif
                    //-----------------------------------------------------------------------------
                    //! Copy assignment.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_CUDA_ONLY auto operator=(IdxCuda const & ) -> IdxCuda & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_CUDA_ONLY /*virtual*/ ~IdxCuda() noexcept = default;

                    //-----------------------------------------------------------------------------
                    //! \return The index of the currently executed thread.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_CUDA_ONLY auto getIdxBlockThread() const
                    -> Vec<TDim>
                    {
                        return offset::getOffsetsVecNd<TDim, UInt>(threadIdx);
                    }
                    //-----------------------------------------------------------------------------
                    //! \return The block index of the currently executed thread.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_CUDA_ONLY auto getIdxGridBlock() const
                    -> Vec<TDim>
                    {
                        return offset::getOffsetsVecNd<TDim, UInt>(blockIdx);
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
            //! The GPU CUDA accelerator index dimension get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DimType<
                accs::cuda::detail::IdxCuda<TDim>>
            {
                using type = TDim;
            };
        }

        namespace idx
        {
            //#############################################################################
            //! The GPU CUDA accelerator block thread index get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetIdx<
                accs::cuda::detail::IdxCuda<TDim>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current thread in the block.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_ACC_CUDA_ONLY static auto getIdx(
                    accs::cuda::detail::IdxCuda<TDim> const & index,
                    TWorkDiv const &)
                -> alpaka::Vec<TDim>
                {
                    return index.getIdxBlockThread();
                }
            };

            //#############################################################################
            //! The GPU CUDA accelerator grid block index get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetIdx<
                accs::cuda::detail::IdxCuda<TDim>,
                origin::Grid,
                unit::Blocks>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current block in the grid.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_ACC_CUDA_ONLY static auto getIdx(
                    accs::cuda::detail::IdxCuda<TDim> const & index,
                    TWorkDiv const &)
                -> alpaka::Vec<TDim>
                {
                    return index.getIdxGridBlock();
                }
            };
        }
    }
}
