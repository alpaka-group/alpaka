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
        //-----------------------------------------------------------------------------
        //! The math traits.
        //-----------------------------------------------------------------------------
        namespace traits
        {
            //#############################################################################
            //! The abs trait.
            //#############################################################################
            template<
                typename T,
                typename TArg,
                typename TSfinae = void>
            struct Abs;
        }

        //-----------------------------------------------------------------------------
        //! Computes the absolute value.
        //!
        //! \tparam T The type of the object specializing Abs.
        //! \tparam TArg The arg type.
        //! \param abs The object specializing Abs.
        //! \param arg The arg.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename T,
            typename TArg>
        ALPAKA_FN_HOST_ACC auto abs(
            T const & abs,
            TArg const & arg)
        -> decltype(
            traits::Abs<
                T,
                TArg>
            ::abs(
                abs,
                arg))
        {
            return traits::Abs<
                T,
                TArg>
            ::abs(
                abs,
                arg);
        }

        namespace traits
        {
            //#############################################################################
            //! The Abs specialization for classes with AbsBase member type.
            //#############################################################################
            template<
                typename T,
                typename TArg>
            struct Abs<
                T,
                TArg,
                typename std::enable_if<
                    std::is_base_of<typename T::AbsBase, typename std::decay<T>::type>::value
                    && (!std::is_same<typename T::AbsBase, typename std::decay<T>::type>::value)>::type>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto abs(
                    T const & abs,
                    TArg const & arg)
                -> decltype(
                    math::abs(
                        static_cast<typename T::AbsBase const &>(abs),
                        arg))
                {
                    // Delegate the call to the base class.
                    return
                        math::abs(
                            static_cast<typename T::AbsBase const &>(abs),
                            arg);
                }
            };
        }
    }
}
