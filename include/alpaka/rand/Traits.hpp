/**
* \file
* Copyright 2015 Benjamin Worpitz
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

#include <utility>                  // std::declval
#include <type_traits>              // std::is_integral, std::is_floating_point, ...

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The random number generation specifics.
    //-----------------------------------------------------------------------------
    namespace rand
    {
        //-----------------------------------------------------------------------------
        //! The random number generator distribution specifics.
        //-----------------------------------------------------------------------------
        namespace distribution
        {
            //-----------------------------------------------------------------------------
            //! The random number generator distribution traits.
            //-----------------------------------------------------------------------------
            namespace traits
            {
                //#############################################################################
                //! The random number float normal distribution get trait.
                //#############################################################################
                template<
                    typename TAcc,
                    typename T,
                    typename TSfinae = void>
                struct CreateNormalReal;

                //#############################################################################
                //! The random number float uniform distribution get trait.
                //#############################################################################
                template<
                    typename TAcc,
                    typename T,
                    typename TSfinae = void>
                struct CreateUniformReal;

                //#############################################################################
                //! The random number integer uniform distribution get trait.
                //#############################################################################
                template<
                    typename TAcc,
                    typename T,
                    typename TSfinae = void>
                struct CreateUniformUint;
            }

            //-----------------------------------------------------------------------------
            //! \return A normal float distribution with mean 0.0f and standard deviation 1.0f.
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename T,
                typename TAcc>
            ALPAKA_FN_HOST_ACC auto createNormalReal(
                TAcc const & acc)
            -> decltype(
                traits::CreateNormalReal<
                    TAcc,
                    T>
                ::createNormalReal(
                    std::declval<TAcc const &>()))
            {
                static_assert(
                    std::is_floating_point<T>::value,
                    "The value type T has to be a floating point type!");

                return traits::CreateNormalReal<
                    TAcc,
                    T>
                ::createNormalReal(
                    acc);
            }
            //-----------------------------------------------------------------------------
            //! \return A uniform floating point distribution [0.0, 1.0).
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename T,
                typename TAcc>
            ALPAKA_FN_HOST_ACC auto createUniformReal(
                TAcc const & acc)
            -> decltype(
                traits::CreateUniformReal<
                    TAcc,
                    T>
                ::createUniformReal(
                    std::declval<TAcc const &>()))
            {
                static_assert(
                    std::is_floating_point<T>::value,
                    "The value type T has to be a floating point type!");

                return traits::CreateUniformReal<
                    TAcc,
                    T>
                ::createUniformReal(
                    acc);
            }
            //-----------------------------------------------------------------------------
            //! \return A uniform integer distribution [0, UINT_MAX].
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename T,
                typename TAcc>
            ALPAKA_FN_HOST_ACC auto createUniformUint(
                TAcc const & acc)
            -> decltype(
                traits::CreateUniformUint<
                    TAcc,
                    T>
                ::createUniformUint(
                    std::declval<TAcc const &>()))
            {
                static_assert(
                    std::is_integral<T>::value && std::is_unsigned<T>::value,
                    "The value type T has to be a unsigned integral type!");

                return traits::CreateUniformUint<
                    TAcc,
                    T>
                ::createUniformUint(
                    acc);
            }
        }
        //-----------------------------------------------------------------------------
        //! The random number generator specifics.
        //-----------------------------------------------------------------------------
        namespace generator
        {
            //-----------------------------------------------------------------------------
            //! The random number generator traits.
            //-----------------------------------------------------------------------------
            namespace traits
            {
                //#############################################################################
                //! The random number default generator get trait.
                //#############################################################################
                template<
                    typename TAcc,
                    typename TSfinae = void>
                struct CreateDefault;
            }
            //-----------------------------------------------------------------------------
            //! \return A default random number generator.
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename TAcc>
            ALPAKA_FN_HOST_ACC auto createDefault(
                TAcc const & acc,
                std::uint32_t const & seed,
                std::uint32_t const & subsequence)
            -> decltype(
                traits::CreateDefault<
                    TAcc>
                ::createDefault(
                    std::declval<TAcc const &>(),
                    std::declval<std::uint32_t const &>(),
                    std::declval<std::uint32_t const &>()))
            {
                return traits::CreateDefault<
                    TAcc>
                ::createDefault(
                    acc,
                    seed,
                    subsequence);
            }
        }
    }
}
