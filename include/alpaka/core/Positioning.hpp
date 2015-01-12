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

#include <alpaka/core/Vec.hpp>  // alpaka::vec

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! Defines the origins available for getting extents and indices of kernel executions.
    //-----------------------------------------------------------------------------
    namespace origin
    {
        //#############################################################################
        //! This type is used to get the extent/index relative to the grid.
        //#############################################################################
        struct Grid;
        //#############################################################################
        //! This type is used to get the extent/index relative to a/the current block.
        //#############################################################################
        struct Block;
    }
    //-----------------------------------------------------------------------------
    //! Defines the units available for getting extents and indices of kernel executions.
    //-----------------------------------------------------------------------------
    namespace unit
    {
        //#############################################################################
        //! This type is used to get the extent/index in units of kernels.
        //#############################################################################
        struct Kernels;
        //#############################################################################
        //! This type is used to get the extent/index in units of blocks.
        //#############################################################################
        struct Blocks;
    }
    //-----------------------------------------------------------------------------
    //! Defines the dimensions available for extents sizes and indices of kernel executions.
    //-----------------------------------------------------------------------------
    namespace dim
    {
        //#############################################################################
        //! This type is used to get the extent/index linearized.
        //#############################################################################
        struct Linear;
        //#############################################################################
        //! This type is used to get the extent/index 3-dimensional.
        //#############################################################################
        struct D3;
    }

    using namespace origin;
    using namespace unit;
    using namespace dim;

    //-----------------------------------------------------------------------------
    //! Defines implementation details that should not be used directly by the user.
    //-----------------------------------------------------------------------------
    namespace detail
    {
        //#############################################################################
        //! The trait for retrieving the return type of the getExtent functions depending on the dimensionality.
        //#############################################################################
        template<class TDimensionality>
        struct DimToRetType;

        template<>
        struct DimToRetType<dim::D3>
        {
            using type = vec<3u>;
        };

        template<>
        struct DimToRetType<dim::Linear>
        {
            using type = vec<3u>::Value;
        };
    }
}
