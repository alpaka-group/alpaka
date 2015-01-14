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

#include <alpaka/interfaces/WorkExtent.hpp> // IWorkExtent

namespace alpaka
{
    namespace cuda
    {
        namespace detail
        {
            //#############################################################################
            //! The CUDA accelerator work extent.
            //#############################################################################
            class WorkExtentCuda
            {
            public:
                //-----------------------------------------------------------------------------
                //! Default-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY WorkExtentCuda() = default;
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY WorkExtentCuda(WorkExtentCuda const &) = default;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY WorkExtentCuda(WorkExtentCuda &&) = default;
                //-----------------------------------------------------------------------------
                //! Copy assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY WorkExtentCuda & operator=(WorkExtentCuda const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY /*virtual*/ ~WorkExtentCuda() noexcept = default;

                //-----------------------------------------------------------------------------
                //! \return The grid dimensions of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY Vec<3u> getExtentGridBlocks() const
                {
                    return {gridDim.x, gridDim.y, gridDim.z};
                }
                //-----------------------------------------------------------------------------
                //! \return The block dimensions of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY Vec<3u> getExtentBlockKernels() const
                {
                    return {blockDim.x, blockDim.y, blockDim.z};
                }
            };
            using InterfacedWorkExtentCuda = alpaka::IWorkExtent<WorkExtentCuda>;
        }
    }
}
