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
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename T,
            typename Ty,
            typename Tx>
        ALPAKA_FN_HOST_ACC auto atan2(
            T const & atan2,
            Ty const & y,
            Tx const & x)
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
            return
                traits::Atan2<
                    T,
                    Ty,
                    Tx>
                ::atan2(
                    atan2,
                    y,
                    x);
        }

        namespace traits
        {
            //#############################################################################
            //! The Atan2 specialization for classes with Atan2Base member type.
            //#############################################################################
            template<
                typename T,
                typename Ty,
                typename Tx>
            struct Atan2<
                T,
                Ty,
                Tx,
                typename std::enable_if<
                    std::is_base_of<typename T::Atan2Base, typename std::decay<T>::type>::value
                    && (!std::is_same<typename T::Atan2Base, typename std::decay<T>::type>::value)>::type>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto atan2(
                    T const & atan2,
                    Ty const & y,
                    Tx const & x)
                -> decltype(
                    math::atan2(
                        static_cast<typename T::Atan2Base const &>(atan2),
                        y,
                        x))
                {
                    // Delegate the call to the base class.
                    return
                        math::atan2(
                            static_cast<typename T::Atan2Base const &>(atan2),
                            y,
                            x);
                }
            };
        }
    }
}
