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

#include <alpaka/interfaces/WorkSize.hpp>   // IWorkSize

namespace alpaka
{
    namespace cuda
    {
        namespace detail
        {
            //#############################################################################
            //! This class holds the implementation details for the work sizes of the CUDA accelerator.
            //#############################################################################
            class WorkSizeCuda
            {
            public:
                //-----------------------------------------------------------------------------
                //! Default-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC WorkSizeCuda() = default;
                //-----------------------------------------------------------------------------
                //! Copy-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC WorkSizeCuda(WorkSizeCuda const &) = default;
                //-----------------------------------------------------------------------------
                //! Move-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC WorkSizeCuda(WorkSizeCuda &&) = default;
                //-----------------------------------------------------------------------------
                //! Assignment-operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC WorkSizeCuda & operator=(WorkSizeCuda const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST ~WorkSizeCuda() noexcept = default;

                //-----------------------------------------------------------------------------
                //! \return The grid dimensions of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC vec<3u> getSizeGridBlocks() const
                {
//#ifdef __CUDA_ARCH__
                    return {gridDim.x, gridDim.y, gridDim.z};
//#else
//                    throw std::logic_error("WorkSizeCuda can not be used in non-CUDA Code!");
//#endif
                }
                //-----------------------------------------------------------------------------
                //! \return The block dimensions of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC vec<3u> getSizeBlockKernels() const
                {
//#ifdef __CUDA_ARCH__
                    return {blockDim.x, blockDim.y, blockDim.z};
//#else
//                    throw std::logic_error("WorkSizeCuda can not be used in non-CUDA Code!");
//#endif
                }
            };
            using TInterfacedWorkSize = alpaka::IWorkSize<WorkSizeCuda>;
        }
    }
}
