/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/core/Dict.hpp>
#include <alpaka/core/Tag.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <functional>

using namespace alpaka;

TEST_CASE("dict mutate entry", "")
{
    ALPAKA_TAG(a);
    ALPAKA_TAG(b);
    ALPAKA_TAG(c);

    static_assert(std::is_same<decltype(a), decltype(a)>::value);
    static_assert(!std::is_same<decltype(a), decltype(b)>::value);

    int aValue = 42;
    int bValue = 43;
    Dict dictionary = Dict{makeTuple(DictEntry(a, std::ref(aValue)), DictEntry(b, bValue))};

    static_assert(hasTag(dictionary, a));
    static_assert(hasTag(dictionary, b));
    static_assert(!hasTag(dictionary, c));

    static_assert(getIdx(dictionary, a) == 0);
    CHECK(getTag(dictionary, a) == 42);
    CHECK(getTag(dictionary, b) == 43);

    aValue++;
    bValue += 2;

    CHECK(getTag(dictionary, a) == 43);
    CHECK(getTag(dictionary, b) == 43);
    CHECK(dictionary[a] == 43);
    CHECK(dictionary[b] == 43);

    DictEntry entry0{a, std::ref(aValue)};
    Dict dictionary2 = {entry0, DictEntry(b, bValue)};
    CHECK(getTag(dictionary2, a) == 43);

    Dict dictionary4 = Dict{makeTuple(DictEntry(a, std::ref(aValue)), DictEntry(b, bValue))};
    dictionary4[a] = 1;
    CHECK(dictionary4[a] == 1);

    Dict dictionary5 = Dict{makeTuple(DictEntry(c, 5))};
    auto joinedDict = joinDict(dictionary4, dictionary5);
    CHECK(joinedDict[a] == 1);
    CHECK(joinedDict[c] == 5);
}
