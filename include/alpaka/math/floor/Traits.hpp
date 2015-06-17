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
            //! The floor trait.
            //#############################################################################
            template<
                typename T,
                typename TArg,
                typename TSfinae = void>
            struct Floor;
        }

        //-----------------------------------------------------------------------------
        //! Computes the largest integer value not greater than arg.
        //!
        //! \tparam T The type of the object specializing Floor.
        //! \tparam TArg The arg type.
        //! \param floor The object specializing Floor.
        //! \param arg The arg.
        //-----------------------------------------------------------------------------
        template<
            typename T,
            typename TArg>
        ALPAKA_FCT_HOST_ACC auto floor(
            T const & floor,
            TArg const & arg)
        -> decltype(
            traits::Floor<
                T,
                TArg>
            ::floor(
                floor,
                arg))
        {
            return traits::Floor<
                T,
                TArg>
            ::floor(
                floor,
                arg);
        }
    }
}
