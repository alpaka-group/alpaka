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
            //! The trunc trait.
            //#############################################################################
            template<
                typename T,
                typename TArg,
                typename TSfinae = void>
            struct Trunc;
        }

        //-----------------------------------------------------------------------------
        //! Computes the nearest integer not greater in magnitude than arg.
        //!
        //! \tparam T The type of the object specializing Trunc.
        //! \tparam TArg The arg type.
        //! \param trunc The object specializing Trunc.
        //! \param arg The arg.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename T,
            typename TArg>
        ALPAKA_FN_HOST_ACC auto trunc(
            T const & trunc,
            TArg const & arg)
        -> decltype(
            traits::Trunc<
                T,
                TArg>
            ::trunc(
                trunc,
                arg))
        {
            return traits::Trunc<
                T,
                TArg>
            ::trunc(
                trunc,
                arg);
        }

        namespace traits
        {
            //#############################################################################
            //! The Trunc specialization for classes with TruncBase member type.
            //#############################################################################
            template<
                typename T,
                typename TArg>
            struct Trunc<
                T,
                TArg,
                typename std::enable_if<
                    std::is_base_of<typename T::TruncBase, typename std::decay<T>::type>::value
                    && (!std::is_same<typename T::TruncBase, typename std::decay<T>::type>::value)>::type>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto trunc(
                    T const & trunc,
                    TArg const & arg)
                -> decltype(
                    math::trunc(
                        static_cast<typename T::TruncBase const &>(trunc),
                        arg))
                {
                    // Delegate the call to the base class.
                    return
                        math::trunc(
                            static_cast<typename T::TruncBase const &>(trunc),
                            arg);
                }
            };
        }
    }
}
