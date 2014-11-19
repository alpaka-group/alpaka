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

#include <alpaka/interfaces/Index.hpp>  // IIndex

namespace alpaka
{
    namespace cuda
    {
        namespace detail
        {
            //#############################################################################
            //! This class holds the implementation details for the indexing of the CUDA accelerator.
            //#############################################################################
            class IndexCuda
            {
            public:
                //-----------------------------------------------------------------------------
                //! Default-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC IndexCuda() = default;
                //-----------------------------------------------------------------------------
                //! Copy-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC IndexCuda(IndexCuda const & other) = default;
                //-----------------------------------------------------------------------------
                //! Move-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC IndexCuda(IndexCuda && other) = default;
                //-----------------------------------------------------------------------------
                //! Assignment-operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC IndexCuda & operator=(IndexCuda const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST ~IndexCuda() noexcept = default;

                //-----------------------------------------------------------------------------
                //! \return The index of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC vec<3u> getIdxBlockKernel() const
                {
                    return {threadIdx.x, threadIdx.y, threadIdx.z};
                }
                //-----------------------------------------------------------------------------
                //! \return The block index of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC vec<3u> getIdxGridBlock() const
                {
                    return {blockIdx.x, blockIdx.y, blockIdx.z};
                }
            };
            using TInterfacedIndex = alpaka::detail::IIndex<IndexCuda>;
        }
    }
}
