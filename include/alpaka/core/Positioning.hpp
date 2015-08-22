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

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! Defines the origins available for getting extents and indices of kernel executions.
    //-----------------------------------------------------------------------------
    namespace origin
    {
        //#############################################################################
        //! This type is used to get the extents/indices relative to the grid.
        //#############################################################################
        struct Grid;
        //#############################################################################
        //! This type is used to get the extents/indices relative to a/the current block.
        //#############################################################################
        struct Block;
    }
    //-----------------------------------------------------------------------------
    //! Defines the units available for getting extents and indices of kernel executions.
    //-----------------------------------------------------------------------------
    namespace unit
    {
        //#############################################################################
        //#############################################################################
        //! This type is used to get the extents/indices in units of threads.
        //#############################################################################
        struct Threads;
        //#############################################################################
        //! This type is used to get the extents/indices in units of blocks.
        //#############################################################################
        struct Blocks;
    }

    using namespace origin;
    using namespace unit;
}
