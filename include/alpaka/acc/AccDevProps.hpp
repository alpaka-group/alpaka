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

#include <alpaka/core/Vec.hpp>      // Vec
#include <alpaka/core/Common.hpp>   // ALPAKA_FN_HOST_ACC

#include <vector>                   // std::vector
#include <string>                   // std::string

namespace alpaka
{
    namespace acc
    {
        //#############################################################################
        //! The acceleration properties on a device.
        //#############################################################################
        template<
            typename TDim,
            typename TSize>
        struct AccDevProps
        {
            //-----------------------------------------------------------------------------
            //! Default-constructor
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST AccDevProps(
                TSize const & uiMultiProcessorCount,
                TSize const & uiBlockThreadsCountMax,
                Vec<TDim, TSize> const & vuiBlockThreadExtentsMax,
                Vec<TDim, TSize> const & vuiGridBlockExtentsMax) :
                    m_uiMultiProcessorCount(uiMultiProcessorCount),
                    m_uiBlockThreadsCountMax(uiBlockThreadsCountMax),
                    m_vuiBlockThreadExtentsMax(vuiBlockThreadExtentsMax),
                    m_vuiGridBlockExtentsMax(vuiGridBlockExtentsMax)
            {}

            TSize m_uiMultiProcessorCount;                  //!< The number of multiprocessors.
            TSize m_uiBlockThreadsCountMax;                 //!< The maximum number of threads in a block.
            Vec<TDim, TSize> m_vuiBlockThreadExtentsMax;    //!< The maximum number of threads in each dimension of a block.
            Vec<TDim, TSize> m_vuiGridBlockExtentsMax;      //!< The maximum number of blocks in each dimension of the grid.
            //TSize m_uiSharedMemSizeBytes;                 //!< Size of the available block shared memory in bytes.
        };
    }
}
