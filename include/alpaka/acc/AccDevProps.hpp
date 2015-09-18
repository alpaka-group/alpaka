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
                Vec<TDim, TSize> const & gridBlockExtentMax,
                TSize const & gridBlockCountMax,
                Vec<TDim, TSize> const & blockThreadExtentMax,
                TSize const & blockThreadCountMax,
                Vec<TDim, TSize> const & threadElemExtentMax,
                TSize const & threadElemCountMax) :
                    m_multiProcessorCount(multiProcessorCount),
                    m_gridBlockExtentMax(gridBlockExtentMax),
                    m_gridBlockCountMax(gridBlockCountMax),
                    m_blockThreadExtentMax(blockThreadExtentMax),
                    m_blockThreadCountMax(blockThreadCountMax),
                    m_threadElemExtentMax(threadElemExtentMax),
                    m_threadElemCountMax(threadElemCountMax)
            {}

            TSize m_multiProcessorCount;                //!< The number of multiprocessors.
            Vec<TDim, TSize> m_gridBlockExtentMax;      //!< The maximum number of blocks in each dimension of the grid.
            TSize m_gridBlockCountMax;                  //!< The maximum number of blocks in a grid.
            Vec<TDim, TSize> m_blockThreadExtentMax;    //!< The maximum number of threads in each dimension of a block.
            TSize m_blockThreadCountMax;                //!< The maximum number of threads in a block.
            Vec<TDim, TSize> m_threadElemExtentMax;     //!< The maximum number of elements in each dimension of a thread.
            TSize m_threadElemCountMax;                 //!< The maximum number of elements in a threads.
        };
    }
}
