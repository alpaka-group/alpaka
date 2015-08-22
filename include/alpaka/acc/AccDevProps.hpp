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

#include <alpaka/vec/Vec.hpp>       // Vec
#include <alpaka/core/Common.hpp>   // ALPAKA_FN_HOST_ACC

#include <vector>                   // std::vector
#include <string>                   // std::string

namespace alpaka
{
    namespace acc
    {
        //#############################################################################
        //! The acceleration properties on a device.
        //
        // \TODO:
        //  TSize m_maxClockFrequencyHz;            //!< Maximum clock frequency of the device in Hz.
        //  TSize m_sharedMemSizeBytes;             //!< Size of the available block shared memory in bytes.
        //#############################################################################
        template<
            typename TDim,
            typename TSize>
        struct AccDevProps
        {
            //-----------------------------------------------------------------------------
            //! Default-constructor
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST AccDevProps(
                TSize const & multiProcessorCount,
                TSize const & blockThreadsCountMax,
                Vec<TDim, TSize> const & blockThreadExtentsMax,
                Vec<TDim, TSize> const & gridBlockExtentsMax) :
                    m_multiProcessorCount(multiProcessorCount),
                    m_blockThreadsCountMax(blockThreadsCountMax),
                    m_blockThreadExtentsMax(blockThreadExtentsMax),
                    m_gridBlockExtentsMax(gridBlockExtentsMax)
            {}

            TSize m_multiProcessorCount;                //!< The number of multiprocessors.
            TSize m_blockThreadsCountMax;               //!< The maximum number of threads in a block.
            Vec<TDim, TSize> m_blockThreadExtentsMax;   //!< The maximum number of threads in each dimension of a block.
            Vec<TDim, TSize> m_gridBlockExtentsMax;     //!< The maximum number of blocks in each dimension of the grid.
        };
    }
}
