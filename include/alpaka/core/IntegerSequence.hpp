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

#include <boost/predef.h>   // workarounds

#include <type_traits>      // std::is_integral
#include <cstddef>          // std::size_t

namespace alpaka
{
    namespace detail
    {
        // This could be replaced with c++14 std::integer_sequence if we raise the minimum.
        template<
            class T,
            T... TVals>
        struct integer_sequence
        {
            static_assert(std::is_integral<T>::value, "integer_sequence<T, I...> requires T to be an integral type.");

            typedef integer_sequence<T, TVals...> type;
            typedef T value_type;

            static auto size() noexcept
            -> std::size_t
            {
                return (sizeof...(TVals));
            }
        };

        template<bool TbNegativeSize, bool TbIsBegin, class T, T TuiBegin, class TIntCon, class TIntSeq>
        struct make_integer_sequence_helper
        {
            static_assert(!TbNegativeSize, "make_integer_sequence<T, N> requires N to be non-negative.");
        };
        template<class T, T TuiBegin, T... TVals>
        struct make_integer_sequence_helper<false, true, T, TuiBegin, std::integral_constant<T, TuiBegin>, integer_sequence<T, TVals...> > :
            integer_sequence<T, TVals...>
        {};
        template<class T, T TuiBegin, T TIdx, T... TVals>
        struct make_integer_sequence_helper<false, false, T, TuiBegin, std::integral_constant<T, TIdx>, integer_sequence<T, TVals...> > :
            make_integer_sequence_helper<false, TIdx == (TuiBegin+1), T, TuiBegin, std::integral_constant<T, TIdx - 1>, integer_sequence<T, TIdx - 1, TVals...> >
        {};

        // Bug: https://connect.microsoft.com/VisualStudio/feedback/details/1085630/template-alias-internal-error-in-the-compiler-because-of-tmp-c-integer-sequence-for-c-11
#if (BOOST_COMP_MSVC) && (BOOST_COMP_MSVC < BOOST_VERSION_NUMBER(14, 0, 0))
        template<class T, T TuiSize>
        struct make_integer_sequence
        {
            using type = typename make_integer_sequence_helper<(TuiSize < 0), (TuiSize == 0), T, 0, std::integral_constant<T, TuiSize>, integer_sequence<T> >::type;
        };
        template<class T, T TuiBegin, T TuiSize>
        struct make_integer_sequence_start
        {
            using type = typename make_integer_sequence_helper<(TuiSize < 0), (TuiSize == 0), T, TuiBegin, std::integral_constant<T, TuiBegin+TuiSize>, integer_sequence<T> >::type;
        };
#else
        template<class T, T TuiSize>
        using make_integer_sequence = typename make_integer_sequence_helper<(TuiSize < 0), (TuiSize == 0), T, 0, std::integral_constant<T, TuiSize>, integer_sequence<T> >::type;

        template<class T, T TuiBegin, T TuiSize>
        using make_integer_sequence_start = typename make_integer_sequence_helper<(TuiSize < 0), (TuiSize == 0), T, TuiBegin, std::integral_constant<T, TuiBegin+TuiSize>, integer_sequence<T> >::type;
#endif

        template<std::size_t... TVals>
        using index_sequence = integer_sequence<std::size_t, TVals...>;

        template<std::size_t TuiSize>
        using make_index_sequence = make_integer_sequence<std::size_t, TuiSize>;

        template<typename... Ts>
        using index_sequence_for = make_index_sequence<sizeof...(Ts)>;
    }
}
