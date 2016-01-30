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

#include <alpaka/alpaka.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(meta)

//#############################################################################
//!
//#############################################################################
template<
    std::size_t TuniqueId = alpaka::meta::uniqueId()>
auto constexpr uniqueIdAsDefaultTemplateParam()
-> std::size_t
{
    return TuniqueId;
};

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(uniqueId)
{
    using IsSetInput =
        std::tuple<
            int,
            float,
            long>;

    auto constexpr a =
        alpaka::meta::uniqueId();
    auto constexpr b =
        alpaka::meta::uniqueId();
    static_assert(
        a != b,
        "alpaka::meta::uniqueId 'a != b' failed!");

    auto constexpr c =
        uniqueIdAsDefaultTemplateParam();
    static_assert(
        a != c,
        "alpaka::meta::uniqueId 'a != c' failed!");
    static_assert(
        b != c,
        "alpaka::meta::uniqueId 'b != c' failed!");
}

BOOST_AUTO_TEST_SUITE_END()
