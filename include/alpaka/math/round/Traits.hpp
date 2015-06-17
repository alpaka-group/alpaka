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
            //! The round trait.
            //#############################################################################
            template<
                typename T,
                typename TArg,
                typename TSfinae = void>
            struct Round;

            //#############################################################################
            //! The round trait.
            //#############################################################################
            template<
                typename T,
                typename TArg,
                typename TSfinae = void>
            struct Lround;

            //#############################################################################
            //! The round trait.
            //#############################################################################
            template<
                typename T,
                typename TArg,
                typename TSfinae = void>
            struct Llround;
        }

        //-----------------------------------------------------------------------------
        //! Computes the nearest integer value to arg (in floating-point format), rounding halfway cases away from zero, regardless of the current rounding mode.
        //!
        //! \tparam T The type of the object specializing Round.
        //! \tparam TArg The arg type.
        //! \param round The object specializing Round.
        //! \param arg The arg.
        //-----------------------------------------------------------------------------
        template<
            typename T,
            typename TArg>
        ALPAKA_FCT_HOST_ACC auto round(
            T const & round,
            TArg const & arg)
        -> decltype(
            traits::Round<
                T,
                TArg>
            ::round(
                round,
                arg))
        {
            return traits::Round<
                T,
                TArg>
            ::round(
                round,
                arg);
        }
        //-----------------------------------------------------------------------------
        //! Computes the nearest integer value to arg (in integer format), rounding halfway cases away from zero, regardless of the current rounding mode.
        //!
        //! \tparam T The type of the object specializing Round.
        //! \tparam TArg The arg type.
        //! \param round The object specializing Round.
        //! \param arg The arg.
        //-----------------------------------------------------------------------------
        template<
            typename T,
            typename TArg>
        ALPAKA_FCT_HOST_ACC auto lround(
            T const & lround,
            TArg const & arg)
        -> long int
        {
            return traits::Lround<
                T,
                TArg>
            ::lround(
                lround,
                arg);
        }
        //-----------------------------------------------------------------------------
        //! Computes the nearest integer value to arg (in integer format), rounding halfway cases away from zero, regardless of the current rounding mode.
        //!
        //! \tparam T The type of the object specializing Round.
        //! \tparam TArg The arg type.
        //! \param round The object specializing Round.
        //! \param arg The arg.
        //-----------------------------------------------------------------------------
        template<
            typename T,
            typename TArg>
        ALPAKA_FCT_HOST_ACC auto llround(
            T const & llround,
            TArg const & arg)
        -> long long int
        {
            return traits::Llround<
                T,
                TArg>
            ::llround(
                llround,
                arg);
        }
    }
}
