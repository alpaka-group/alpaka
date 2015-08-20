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
#include <alpaka/core/Unique.hpp>   // core::unique

#include <boost/predef.h>           // workarounds

#include <type_traits>              // std::is_integral
#include <cstddef>                  // std::size_t

namespace alpaka
{
    namespace core
    {
        namespace detail
        {
            // This could be replaced with c++14 std::integer_sequence if we raise the minimum.
            template<
                typename T,
                T... TVals>
            struct integer_sequence
            {
                static_assert(std::is_integral<T>::value, "integer_sequence<T, I...> requires T to be an integral type.");

                typedef integer_sequence<T, TVals...> type;
                typedef T value_type;

                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto size() noexcept
                -> std::size_t
                {
                    return (sizeof...(TVals));
                }
            };

            template<bool TisSizeNegative, bool TbIsBegin, typename T, T Tbegin, typename TIntCon, typename TIntSeq>
            struct make_integer_sequence_helper
            {
                static_assert(!TisSizeNegative, "make_integer_sequence<T, N> requires N to be non-negative.");
            };
            template<typename T, T Tbegin, T... TVals>
            struct make_integer_sequence_helper<false, true, T, Tbegin, std::integral_constant<T, Tbegin>, integer_sequence<T, TVals...> > :
                integer_sequence<T, TVals...>
            {};
            template<typename T, T Tbegin, T TIdx, T... TVals>
            struct make_integer_sequence_helper<false, false, T, Tbegin, std::integral_constant<T, TIdx>, integer_sequence<T, TVals...> > :
                make_integer_sequence_helper<false, TIdx == (Tbegin+1), T, Tbegin, std::integral_constant<T, TIdx - 1>, integer_sequence<T, TIdx - 1, TVals...> >
            {};

            template<typename T, T Tbegin, T Tsize>
            using make_integer_sequence_offset = typename make_integer_sequence_helper<(Tsize < 0), (Tsize == 0), T, Tbegin, std::integral_constant<T, Tbegin+Tsize>, integer_sequence<T> >::type;

            template<typename T, T Tsize>
            using make_integer_sequence = make_integer_sequence_offset<T, 0u, Tsize>;

            template<std::size_t... TVals>
            using index_sequence = integer_sequence<std::size_t, TVals...>;

            template<typename T, T Tbegin, T Tsize>
            using make_index_sequence_offset = make_integer_sequence_offset<std::size_t, Tbegin, Tsize>;

            template<std::size_t Tsize>
            using make_index_sequence = make_integer_sequence<std::size_t, Tsize>;

            template<typename... Ts>
            using index_sequence_for = make_index_sequence<sizeof...(Ts)>;

            //-----------------------------------------------------------------------------
            //! Checks if the integral values are unique.
            //-----------------------------------------------------------------------------
            template<
                typename T,
                T... Is>
            struct IntegralValuesUnique
            {
                static constexpr bool value = core::detail::unique<std::integral_constant<T, Is>...>::value;
            };

            //-----------------------------------------------------------------------------
            //! Checks if the values in the index sequence are unique.
            //-----------------------------------------------------------------------------
            template<
                typename TIntegerSequence>
            struct IntegerSequenceValuesUnique;
            //-----------------------------------------------------------------------------
            //! Checks if the values in the index sequence are unique.
            //-----------------------------------------------------------------------------
            template<
                typename T,
                T... Is>
            struct IntegerSequenceValuesUnique<
                integer_sequence<T, Is...>>
            {
                static constexpr bool value = IntegralValuesUnique<T, Is...>::value;
            };

            //-----------------------------------------------------------------------------
            //! Checks if the integral values are within the given range.
            //-----------------------------------------------------------------------------
            template<
                typename T,
                T Tmin,
                T Tmax,
                T... Is>
            struct IntegralValuesInRange;
            //-----------------------------------------------------------------------------
            //! Checks if the integral values are within the given range.
            //-----------------------------------------------------------------------------
            template<
                typename T,
                T Tmin,
                T Tmax>
            struct IntegralValuesInRange<
                T,
                Tmin,
                Tmax>
            {
                static constexpr bool value = true;
            };
            //-----------------------------------------------------------------------------
            //! Checks if the integral values are within the given range.
            //-----------------------------------------------------------------------------
            template<
                typename T,
                T Tmin,
                T Tmax,
                T I,
                T... Is>
            struct IntegralValuesInRange<
                T,
                Tmin,
                Tmax,
                I,
                Is...>
            {
                static constexpr bool value = (I >= Tmin) && (I <=Tmax) && IntegralValuesInRange<T, Tmin, Tmax, Is...>::value;
            };

            //-----------------------------------------------------------------------------
            //! Checks if the values in the index sequence are within the given range.
            //-----------------------------------------------------------------------------
            template<
                typename TIntegerSequence,
                typename T,
                T Tmin,
                T Tmax>
            struct IntegerSequenceValuesInRange;
            //-----------------------------------------------------------------------------
            //! Checks if the values in the index sequence are within the given range.
            //-----------------------------------------------------------------------------
            template<
                typename T,
                T... Is,
                T Tmin,
                T Tmax>
            struct IntegerSequenceValuesInRange<
                integer_sequence<T, Is...>,
                T,
                Tmin,
                Tmax>
            {
                static constexpr bool value = IntegralValuesInRange<T, Tmin, Tmax, Is...>::value;
            };
        }
    }
}
