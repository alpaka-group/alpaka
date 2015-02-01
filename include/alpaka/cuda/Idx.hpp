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

#include <alpaka/traits/Idx.hpp>  // idx::getIdx

namespace alpaka
{
    namespace cuda
    {
        namespace detail
        {
            //#############################################################################
            //! This CUDA accelerator index provider.
            //#############################################################################
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
                ALPAKA_FCT_ACC_CUDA_ONLY IdxCuda & operator=(IdxCuda const & ) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY /*virtual*/ ~IdxCuda() noexcept = default;

                //-----------------------------------------------------------------------------
                //! \return The index of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY Vec<3u> getIdxBlockKernel() const
                {
                    return Vec<3u>(threadIdx.x, threadIdx.y, threadIdx.z);
                }
                //-----------------------------------------------------------------------------
                //! \return The block index of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY Vec<3u> getIdxGridBlock() const
                {
                    return Vec<3u>(blockIdx.x, blockIdx.y, blockIdx.z);
                }
            };
        }
    }

    namespace traits
    {
        namespace idx
        {
            //#############################################################################
            //! The CUDA accelerator 3D block kernel index get trait specialization.
            //#############################################################################
            template<>
            struct GetIdx<
                cuda::detail::IdxCuda,
                origin::Block,
                unit::Kernels,
                alpaka::dim::Dim3>
            {
                //-----------------------------------------------------------------------------
                //! \return The 3-dimensional index of the current kernel in the block.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_ACC_CUDA_ONLY static alpaka::dim::DimToVecT<alpaka::dim::Dim3> getIdx(
                    cuda::detail::IdxCuda const & index,
                    TWorkDiv const &)
                {
                    return index.getIdxBlockKernel();
                }
            };

            //#############################################################################
            //! The CUDA accelerator 3D grid block index get trait specialization.
            //#############################################################################
            template<>
            struct GetIdx<
                cuda::detail::IdxCuda,
                origin::Grid,
                unit::Blocks,
                alpaka::dim::Dim3>
            {
                //-----------------------------------------------------------------------------
                //! \return The 3-dimensional index of the current block in the grid.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_ACC_CUDA_ONLY static alpaka::dim::DimToVecT<alpaka::dim::Dim3> getIdx(
                    cuda::detail::IdxCuda const & index,
                    TWorkDiv const &)
                {
                    return index.getIdxGridBlock();
                }
            };
        }
    }
}
