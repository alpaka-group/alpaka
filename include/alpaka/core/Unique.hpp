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

#include <utility>                          // std::is_base_of

namespace alpaka
{
    namespace core
    {
        namespace detail
        {
            //#############################################################################
            //! Empty dependent type.
            //#############################################################################
            template<
                typename T>
            struct empty
            {};

            //#############################################################################
            //! Trait that tells if the parameter pack contains only unique (no equal) types.
            //#############################################################################
            template<
                typename... Ts>
            struct unique;
            //#############################################################################
            //! Trait that tells if the parameter pack contains only unique (no equal) types.
            //#############################################################################
            template<>
            struct unique<>
            {
                static constexpr bool value = true;
            };
            //#############################################################################
            //! Trait that tells if the parameter pack contains only unique (no equal) types.
            // 
            // Based on code by Roland Bock: https://gist.github.com/rbock/ad8eedde80c060132a18
            // Linearly inherits from empty<T> and checks if it has already inherited from this type.
            //#############################################################################
            template<
                typename T,
                typename... Ts>
            struct unique<T, Ts...> :
                public unique<Ts...>,
                public empty<T>
            {
                using base = unique<Ts...>;

                static constexpr bool value = base::value && !std::is_base_of<empty<T>, base>::value;
            };
        }
    }
}
