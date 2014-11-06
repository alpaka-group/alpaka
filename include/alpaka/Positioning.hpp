/**
* Copyright 2014 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of of either the GNU General Public License or
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

#include <alpaka/Vec.hpp>   // alpaka::vec<3>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! Defines the origins available for getting sizes and indices of kernel executions.
    //-----------------------------------------------------------------------------
    namespace origin
    {
        //#############################################################################
        //! This type is used to get the size/index relative to the grid.
        //#############################################################################
        struct Grid;
        //#############################################################################
        //! This type is used to get the size/index relative to a/the current block.
        //#############################################################################
        struct Block;
    }
    //-----------------------------------------------------------------------------
    //! Defines the units available for getting sizes and indices of kernel executions.
    //-----------------------------------------------------------------------------
    namespace unit
    {
        //#############################################################################
        //! This type is used to get the size/index in units of kernels.
        //#############################################################################
        struct Kernels;
        //#############################################################################
        //! This type is used to get the size/index in units of blocks.
        //#############################################################################
        struct Blocks;
    }
    //-----------------------------------------------------------------------------
    //! Defines the dimensions available for getting sizes and indices of kernel executions.
    //-----------------------------------------------------------------------------
    namespace dim
    {
        //#############################################################################
        //! This type is used to get the size/index linearized.
        //#############################################################################
        struct Linear;
        //#############################################################################
        //! This type is used to get the size/index 3-dimensional.
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
        //! The trait for retrieving the return type of the getSize functions depending on the dimensionality.
        //#############################################################################
        template<class TDimensionality>
        struct DimToRetType;

        template<>
        struct DimToRetType<dim::D3>
        {
            using type = vec<3>;
        };

        template<>
        struct DimToRetType<dim::Linear>
        {
            using type = std::uint32_t;
        };
    }
}