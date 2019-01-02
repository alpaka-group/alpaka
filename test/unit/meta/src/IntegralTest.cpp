/**
 * \file
 * Copyright 2017-2019 Benjamin Worpitz
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

#include <type_traits>

//-----------------------------------------------------------------------------
TEST_CASE("isIntegralSupersetTrue", "[meta]")
{
    // unsigned - unsigned
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::uint64_t, std::uint64_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::uint64_t, std::uint32_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::uint64_t, std::uint16_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::uint64_t, std::uint8_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        alpaka::meta::IsIntegralSuperset<std::uint32_t, std::uint32_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::uint32_t, std::uint16_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::uint32_t, std::uint8_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        alpaka::meta::IsIntegralSuperset<std::uint16_t, std::uint16_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::uint16_t, std::uint8_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        alpaka::meta::IsIntegralSuperset<std::uint8_t, std::uint8_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    // signed - signed
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::int64_t, std::int64_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::int64_t, std::int32_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::int64_t, std::int16_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::int64_t, std::int8_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        alpaka::meta::IsIntegralSuperset<std::int32_t, std::int32_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::int32_t, std::int16_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::int32_t, std::int8_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        alpaka::meta::IsIntegralSuperset<std::int16_t, std::int16_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::int16_t, std::int8_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        alpaka::meta::IsIntegralSuperset<std::int8_t, std::int8_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    // unsigned - signed
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::uint64_t, std::int32_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::uint64_t, std::int16_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::uint64_t, std::int8_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        alpaka::meta::IsIntegralSuperset<std::uint32_t, std::int16_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::uint32_t, std::int8_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        alpaka::meta::IsIntegralSuperset<std::uint16_t, std::int8_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    // signed - unsigned
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::int64_t, std::uint32_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::int64_t, std::uint16_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::int64_t, std::uint8_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        alpaka::meta::IsIntegralSuperset<std::int32_t, std::uint16_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        alpaka::meta::IsIntegralSuperset<std::int32_t, std::uint8_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        alpaka::meta::IsIntegralSuperset<std::int16_t, std::uint8_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
}

//-----------------------------------------------------------------------------
TEST_CASE("isIntegralSupersetNoIntegral", "[meta]")
{
    static_assert(
        !alpaka::meta::IsIntegralSuperset<float, std::uint8_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::uint64_t, double>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
}

//-----------------------------------------------------------------------------
TEST_CASE("isIntegralSupersetFalse", "[meta]")
{
    // unsigned - unsigned
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::uint8_t, std::uint64_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::uint8_t, std::uint32_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::uint8_t, std::uint16_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::uint16_t, std::uint64_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::uint16_t, std::uint32_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::uint32_t, std::uint64_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    // signed - signed
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::int8_t, std::int64_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::int8_t, std::int32_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::int8_t, std::int16_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::int16_t, std::int64_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::int16_t, std::int32_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::int32_t, std::int64_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    // unsigned - signed
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::uint64_t, std::int64_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::uint32_t, std::int64_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::uint32_t, std::int32_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::uint16_t, std::int64_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::uint16_t, std::int32_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::uint16_t, std::int16_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::uint8_t, std::int64_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::uint8_t, std::int32_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::uint8_t, std::int16_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::uint8_t, std::int8_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    // signed - unsigned
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::int64_t, std::uint64_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::int32_t, std::uint64_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::int32_t, std::uint32_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::int16_t, std::uint64_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::int16_t, std::uint32_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::int16_t, std::uint16_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");

    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::int8_t, std::uint64_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::int8_t, std::uint32_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::int8_t, std::uint16_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
    static_assert(
        !alpaka::meta::IsIntegralSuperset<std::int8_t, std::uint8_t>::value,
        "alpaka::meta::IsIntegralSuperset failed!");
}

//-----------------------------------------------------------------------------
TEST_CASE("higherMax", "[meta]")
{
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int8_t, std::int8_t>, std::int8_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int8_t, std::uint8_t>, std::uint8_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int8_t, std::int16_t>, std::int16_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int8_t, std::uint16_t>, std::uint16_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int8_t, std::int32_t>, std::int32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int8_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int8_t, std::int64_t>, std::int64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int8_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMax failed!");

    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint8_t, std::int8_t>, std::uint8_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint8_t, std::uint8_t>, std::uint8_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint8_t, std::int16_t>, std::int16_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint8_t, std::uint16_t>, std::uint16_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint8_t, std::int32_t>, std::int32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint8_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint8_t, std::int64_t>, std::int64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint8_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMax failed!");

    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int16_t, std::int8_t>, std::int16_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int16_t, std::uint8_t>, std::int16_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int16_t, std::int16_t>, std::int16_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int16_t, std::uint16_t>, std::uint16_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int16_t, std::int32_t>, std::int32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int16_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int16_t, std::int64_t>, std::int64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int16_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMax failed!");

    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint16_t, std::int8_t>, std::uint16_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint16_t, std::uint8_t>, std::uint16_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint16_t, std::int16_t>, std::uint16_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint16_t, std::uint16_t>, std::uint16_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint16_t, std::int32_t>, std::int32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint16_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint16_t, std::int64_t>, std::int64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint16_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMax failed!");

    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int32_t, std::int8_t>, std::int32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int32_t, std::uint8_t>, std::int32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int32_t, std::int16_t>, std::int32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int32_t, std::uint16_t>, std::int32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int32_t, std::int32_t>, std::int32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int32_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int32_t, std::int64_t>, std::int64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int32_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMax failed!");

    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint32_t, std::int8_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint32_t, std::uint8_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint32_t, std::int16_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint32_t, std::uint16_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint32_t, std::int32_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint32_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint32_t, std::int64_t>, std::int64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint32_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMax failed!");

    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int64_t, std::int8_t>, std::int64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int64_t, std::uint8_t>, std::int64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int64_t, std::int16_t>, std::int64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int64_t, std::uint16_t>, std::int64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int64_t, std::int32_t>, std::int64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int64_t, std::uint32_t>, std::int64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int64_t, std::int64_t>, std::int64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::int64_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMax failed!");

    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint64_t, std::int8_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint64_t, std::uint8_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint64_t, std::int16_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint64_t, std::uint16_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint64_t, std::int32_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint64_t, std::uint32_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint64_t, std::int64_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMax failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMax<std::uint64_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMax failed!");
}

//-----------------------------------------------------------------------------
TEST_CASE("lowerMax", "[meta]")
{
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int8_t, std::int8_t>, std::int8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int8_t, std::uint8_t>, std::int8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int8_t, std::int16_t>, std::int8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int8_t, std::uint16_t>, std::int8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int8_t, std::int32_t>, std::int8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int8_t, std::uint32_t>, std::int8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int8_t, std::int64_t>, std::int8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int8_t, std::uint64_t>, std::int8_t>::value,
        "alpaka::meta::LowerMax failed!");

    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint8_t, std::int8_t>, std::int8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint8_t, std::uint8_t>, std::uint8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint8_t, std::int16_t>, std::uint8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint8_t, std::uint16_t>, std::uint8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint8_t, std::int32_t>, std::uint8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint8_t, std::uint32_t>, std::uint8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint8_t, std::int64_t>, std::uint8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint8_t, std::uint64_t>, std::uint8_t>::value,
        "alpaka::meta::LowerMax failed!");

    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int16_t, std::int8_t>, std::int8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int16_t, std::uint8_t>, std::uint8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int16_t, std::int16_t>, std::int16_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int16_t, std::uint16_t>, std::int16_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int16_t, std::int32_t>, std::int16_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int16_t, std::uint32_t>, std::int16_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int16_t, std::int64_t>, std::int16_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int16_t, std::uint64_t>, std::int16_t>::value,
        "alpaka::meta::LowerMax failed!");

    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint16_t, std::int8_t>, std::int8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint16_t, std::uint8_t>, std::uint8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint16_t, std::int16_t>, std::int16_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint16_t, std::uint16_t>, std::uint16_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint16_t, std::int32_t>, std::uint16_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint16_t, std::uint32_t>, std::uint16_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint16_t, std::int64_t>, std::uint16_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint16_t, std::uint64_t>, std::uint16_t>::value,
        "alpaka::meta::LowerMax failed!");

    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int32_t, std::int8_t>, std::int8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int32_t, std::uint8_t>, std::uint8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int32_t, std::int16_t>, std::int16_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int32_t, std::uint16_t>, std::uint16_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int32_t, std::int32_t>, std::int32_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int32_t, std::uint32_t>, std::int32_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int32_t, std::int64_t>, std::int32_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int32_t, std::uint64_t>, std::int32_t>::value,
        "alpaka::meta::LowerMax failed!");

    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint32_t, std::int8_t>, std::int8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint32_t, std::uint8_t>, std::uint8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint32_t, std::int16_t>, std::int16_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint32_t, std::uint16_t>, std::uint16_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint32_t, std::int32_t>, std::int32_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint32_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint32_t, std::int64_t>, std::uint32_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint32_t, std::uint64_t>, std::uint32_t>::value,
        "alpaka::meta::LowerMax failed!");

    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int64_t, std::int8_t>, std::int8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int64_t, std::uint8_t>, std::uint8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int64_t, std::int16_t>, std::int16_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int64_t, std::uint16_t>, std::uint16_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int64_t, std::int32_t>, std::int32_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int64_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int64_t, std::int64_t>, std::int64_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::int64_t, std::uint64_t>, std::int64_t>::value,
        "alpaka::meta::LowerMax failed!");

    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint64_t, std::int8_t>, std::int8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint64_t, std::uint8_t>, std::uint8_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint64_t, std::int16_t>, std::int16_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint64_t, std::uint16_t>, std::uint16_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint64_t, std::int32_t>, std::int32_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint64_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint64_t, std::int64_t>, std::int64_t>::value,
        "alpaka::meta::LowerMax failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMax<std::uint64_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::LowerMax failed!");
}

//-----------------------------------------------------------------------------
TEST_CASE("higherMin", "[meta]")
{
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int8_t, std::int8_t>, std::int8_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int8_t, std::uint8_t>, std::uint8_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int8_t, std::int16_t>, std::int8_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int8_t, std::uint16_t>, std::uint16_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int8_t, std::int32_t>, std::int8_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int8_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int8_t, std::int64_t>, std::int8_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int8_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMin failed!");

    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint8_t, std::int8_t>, std::uint8_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint8_t, std::uint8_t>, std::uint8_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint8_t, std::int16_t>, std::uint8_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint8_t, std::uint16_t>, std::uint16_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint8_t, std::int32_t>, std::uint8_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint8_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint8_t, std::int64_t>, std::uint8_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint8_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMin failed!");

    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int16_t, std::int8_t>, std::int8_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int16_t, std::uint8_t>, std::uint8_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int16_t, std::int16_t>, std::int16_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int16_t, std::uint16_t>, std::uint16_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int16_t, std::int32_t>, std::int16_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int16_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int16_t, std::int64_t>, std::int16_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int16_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMin failed!");

    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint16_t, std::int8_t>, std::uint16_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint16_t, std::uint8_t>, std::uint16_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint16_t, std::int16_t>, std::uint16_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint16_t, std::uint16_t>, std::uint16_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint16_t, std::int32_t>, std::uint16_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint16_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint16_t, std::int64_t>, std::uint16_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint16_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMin failed!");

    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int32_t, std::int8_t>, std::int8_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int32_t, std::uint8_t>, std::uint8_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int32_t, std::int16_t>, std::int16_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int32_t, std::uint16_t>, std::uint16_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int32_t, std::int32_t>, std::int32_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int32_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int32_t, std::int64_t>, std::int32_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int32_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMin failed!");

    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint32_t, std::int8_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint32_t, std::uint8_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint32_t, std::int16_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint32_t, std::uint16_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint32_t, std::int32_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint32_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint32_t, std::int64_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint32_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMin failed!");

    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int64_t, std::int8_t>, std::int8_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int64_t, std::uint8_t>, std::uint8_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int64_t, std::int16_t>, std::int16_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int64_t, std::uint16_t>, std::uint16_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int64_t, std::int32_t>, std::int32_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int64_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int64_t, std::int64_t>, std::int64_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::int64_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMin failed!");

    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint64_t, std::int8_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint64_t, std::uint8_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint64_t, std::int16_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint64_t, std::uint16_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint64_t, std::int32_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint64_t, std::uint32_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint64_t, std::int64_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMin failed!");
    static_assert(
        std::is_same<alpaka::meta::HigherMin<std::uint64_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::HigherMin failed!");
}
//-----------------------------------------------------------------------------
TEST_CASE("lowerMin", "[meta]")
{
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int8_t, std::int8_t>, std::int8_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int8_t, std::uint8_t>, std::int8_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int8_t, std::int16_t>, std::int16_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int8_t, std::uint16_t>, std::int8_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int8_t, std::int32_t>, std::int32_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int8_t, std::uint32_t>, std::int8_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int8_t, std::int64_t>, std::int64_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int8_t, std::uint64_t>, std::int8_t>::value,
        "alpaka::meta::LowerMin failed!");

    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint8_t, std::int8_t>, std::int8_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint8_t, std::uint8_t>, std::uint8_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint8_t, std::int16_t>, std::int16_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint8_t, std::uint16_t>, std::uint16_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint8_t, std::int32_t>, std::int32_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint8_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint8_t, std::int64_t>, std::int64_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint8_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::LowerMin failed!");

    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int16_t, std::int8_t>, std::int16_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int16_t, std::uint8_t>, std::int16_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int16_t, std::int16_t>, std::int16_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int16_t, std::uint16_t>, std::int16_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int16_t, std::int32_t>, std::int32_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int16_t, std::uint32_t>, std::int16_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int16_t, std::int64_t>, std::int64_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int16_t, std::uint64_t>, std::int16_t>::value,
        "alpaka::meta::LowerMin failed!");

    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint16_t, std::int8_t>, std::int8_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint16_t, std::uint8_t>, std::uint16_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint16_t, std::int16_t>, std::int16_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint16_t, std::uint16_t>, std::uint16_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint16_t, std::int32_t>, std::int32_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint16_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint16_t, std::int64_t>, std::int64_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint16_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::LowerMin failed!");

    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int32_t, std::int8_t>, std::int32_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int32_t, std::uint8_t>, std::int32_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int32_t, std::int16_t>, std::int32_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int32_t, std::uint16_t>, std::int32_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int32_t, std::int32_t>, std::int32_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int32_t, std::uint32_t>, std::int32_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int32_t, std::int64_t>, std::int64_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int32_t, std::uint64_t>, std::int32_t>::value,
        "alpaka::meta::LowerMin failed!");

    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint32_t, std::int8_t>, std::int8_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint32_t, std::uint8_t>, std::uint32_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint32_t, std::int16_t>, std::int16_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint32_t, std::uint16_t>, std::uint32_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint32_t, std::int32_t>, std::int32_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint32_t, std::uint32_t>, std::uint32_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint32_t, std::int64_t>, std::int64_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint32_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::LowerMin failed!");

    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int64_t, std::int8_t>, std::int64_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int64_t, std::uint8_t>, std::int64_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int64_t, std::int16_t>, std::int64_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int64_t, std::uint16_t>, std::int64_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int64_t, std::int32_t>, std::int64_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int64_t, std::uint32_t>, std::int64_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int64_t, std::int64_t>, std::int64_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::int64_t, std::uint64_t>, std::int64_t>::value,
        "alpaka::meta::LowerMin failed!");

    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint64_t, std::int8_t>, std::int8_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint64_t, std::uint8_t>, std::uint64_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint64_t, std::int16_t>, std::int16_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint64_t, std::uint16_t>, std::uint64_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint64_t, std::int32_t>, std::int32_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint64_t, std::uint32_t>, std::uint64_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint64_t, std::int64_t>, std::int64_t>::value,
        "alpaka::meta::LowerMin failed!");
    static_assert(
        std::is_same<alpaka::meta::LowerMin<std::uint64_t, std::uint64_t>, std::uint64_t>::value,
        "alpaka::meta::LowerMin failed!");
}
