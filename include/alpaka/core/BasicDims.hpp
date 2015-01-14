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

#include <boost/mpl/integral_c.hpp>     // boost::mpl::integral_c
#include <cstdint>                      // std::size_t

namespace alpaka
{
    namespace dim
    {
        //-----------------------------------------------------------------------------
        // N dimensions.
        //-----------------------------------------------------------------------------
        template<std::size_t N>
        using Dim = boost::mpl::integral_c<std::size_t, N>;

        //-----------------------------------------------------------------------------
        //! One dimension.
        //-----------------------------------------------------------------------------
        using Dim1 = Dim<1>;
        //-----------------------------------------------------------------------------
        //! Two dimensions.
        //-----------------------------------------------------------------------------
        using Dim2 = Dim<2>;
        //-----------------------------------------------------------------------------
        //! Three dimensions.
        //-----------------------------------------------------------------------------
        using Dim3 = Dim<3>;
    }
}
