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

#include <alpaka/core/Common.hpp>   // ALPAKA_FCT_HOST_ACC

namespace alpaka
{
    namespace math
    {
        namespace traits
        {
            //#############################################################################
            //! The min trait.
            //#############################################################################
            template<
                typename T,
                typename Tx,
                typename Ty,
                typename TSfinae = void>
            struct Min;
        }

        //-----------------------------------------------------------------------------
        //! Returns the smaller of two arguments.
        //! NaNs are treated as missing data (between a NaN and a numeric value, the numeric value is chosen).
        //!
        //! \tparam T The type of the object specializing Min.
        //! \tparam Tx The type of the first argument.
        //! \tparam Ty The type of the second argument.
        //! \param min The object specializing Min.
        //! \param x The first argument.
        //! \param y The second argument.
        //-----------------------------------------------------------------------------
        template<
            typename T,
            typename Tx,
            typename Ty>
        ALPAKA_FCT_HOST_ACC auto min(
            T const & min,
            Tx const & x,
            Ty const & y)
        -> decltype(
            traits::Min<
                T,
                Tx,
                Ty>
            ::min(
                min,
                x,
                y))
        {
            return traits::Min<
                T,
                Tx,
                Ty>
            ::min(
                min,
                x,
                y);
        }
    }
}
