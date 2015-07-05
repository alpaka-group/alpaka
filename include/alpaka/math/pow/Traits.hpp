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

#include <alpaka/core/Common.hpp>   // ALPAKA_FN_HOST_ACC

#include <type_traits>              // std::enable_if, std::is_base_of, std::is_same, std::decay

namespace alpaka
{
    namespace math
    {
        namespace traits
        {
            //#############################################################################
            //! The pow trait.
            //#############################################################################
            template<
                typename T,
                typename TBase,
                typename TExp,
                typename TSfinae = void>
            struct Pow;
        }

        //-----------------------------------------------------------------------------
        //! Computes the value of base raised to the power exp.
        //!
        //! \tparam T The type of the object specializing Pow.
        //! \tparam TBase The base type.
        //! \tparam TExp The exponent type.
        //! \param pow The object specializing Pow.
        //! \param base The base.
        //! \param exp The exponent.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename T,
            typename TBase,
            typename TExp>
        ALPAKA_FN_HOST_ACC auto pow(
            T const & pow,
            TBase const & base,
            TExp const & exp)
        -> decltype(
            traits::Pow<
                T,
                TBase,
                TExp>
            ::pow(
                pow,
                base,
                exp))
        {
            return traits::Pow<
                T,
                TBase,
                TExp>
            ::pow(
                pow,
                base,
                exp);
        }

        namespace traits
        {
            //#############################################################################
            //! The Pow specialization for classes with PowBase member type.
            //#############################################################################
            template<
                typename T,
                typename TBase,
                typename TExp>
            struct Pow<
                T,
                TBase,
                TExp,
                typename std::enable_if<
                    std::is_base_of<typename T::PowBase, typename std::decay<T>::type>::value
                    && (!std::is_same<typename T::PowBase, typename std::decay<T>::type>::value)>::type>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto pow(
                    T const & pow,
                    TBase const & base,
                    TExp const & exp)
                -> decltype(
                    math::pow(
                        static_cast<typename T::PowBase const &>(pow),
                        base,
                        exp))
                {
                    // Delegate the call to the base class.
                    return
                        math::pow(
                            static_cast<typename T::PowBase const &>(pow),
                            base,
                            exp);
                }
            };
        }
    }
}
