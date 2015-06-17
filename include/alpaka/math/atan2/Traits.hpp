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
            //! The atan2 trait.
            //#############################################################################
            template<
                typename T,
                typename Ty,
                typename Tx,
                typename TSfinae = void>
            struct Atan2;
        }

        //-----------------------------------------------------------------------------
        //! Computes the arc tangent of y/x using the signs of arguments to determine the correct quadrant.
        //!
        //! \tparam T The type of the object specializing Atan2.
        //! \tparam Ty The y arg type.
        //! \tparam Tx The x arg type.
        //! \param atan2 The object specializing Atan2.
        //! \param y The y arg.
        //! \param x The x arg.
        //-----------------------------------------------------------------------------
        template<
            typename T,
            typename Ty,
            typename Tx>
        ALPAKA_FCT_HOST_ACC auto atan2(
            T const & atan2,
            Ty const & y,
            Ty const & x)
        -> decltype(
            traits::Atan2<
                T,
                Ty,
                Tx>
            ::atan2(
                atan2,
                y,
                x))
        {
            return traits::Atan2<
                T,
                Ty,
                Tx>
            ::atan2(
                atan2,
                y,
                x);
        }
    }
}
