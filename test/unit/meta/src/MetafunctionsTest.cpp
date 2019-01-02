/**
 * \file
 * Copyright 2015-2019 Benjamin Worpitz
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

#include <alpaka/alpaka.hpp>

#include <catch2/catch.hpp>

#include <tuple>
#include <type_traits>

//-----------------------------------------------------------------------------
TEST_CASE("conjunctionTrue", "[meta]")
{
    using ConjunctionResult =
        alpaka::meta::Conjunction<
            std::true_type,
            std::true_type,
            std::integral_constant<bool, true>
        >;

    static_assert(
        ConjunctionResult::value == true,
        "alpaka::meta::Conjunction failed!");
}

//-----------------------------------------------------------------------------
TEST_CASE("conjunctionFalse", "[meta]")
{
    using ConjunctionResult =
        alpaka::meta::Conjunction<
            std::true_type,
            std::false_type,
            std::integral_constant<bool, true>
        >;

    static_assert(
        ConjunctionResult::value == false,
        "alpaka::meta::Conjunction failed!");
}

//-----------------------------------------------------------------------------
TEST_CASE("disjunctionTrue", "[meta]")
{
    using DisjunctionResult =
        alpaka::meta::Disjunction<
            std::false_type,
            std::true_type,
            std::integral_constant<bool, false>
        >;

    static_assert(
        DisjunctionResult::value == true,
        "alpaka::meta::Disjunction failed!");
}

//-----------------------------------------------------------------------------
TEST_CASE("disjunctionFalse", "[meta]")
{
    using DisjunctionResult =
        alpaka::meta::Disjunction<
            std::false_type,
            std::false_type,
            std::integral_constant<bool, false>
        >;

    static_assert(
        DisjunctionResult::value == false,
        "alpaka::meta::Disjunction failed!");
}

//-----------------------------------------------------------------------------
TEST_CASE("negationFalse", "[meta]")
{
    using NegationResult =
        alpaka::meta::Negation<
            std::true_type
        >;

    using NegationReference =
        std::false_type;

    static_assert(
        std::is_same<
            NegationReference,
            NegationResult
        >::value,
        "alpaka::meta::Negation failed!");
}

//-----------------------------------------------------------------------------
TEST_CASE("negationTrue", "[meta]")
{
    using NegationResult =
        alpaka::meta::Negation<
            std::false_type
        >;

    using NegationReference =
        std::true_type;

    static_assert(
        std::is_same<
            NegationReference,
            NegationResult
        >::value,
        "alpaka::meta::Negation failed!");
}
