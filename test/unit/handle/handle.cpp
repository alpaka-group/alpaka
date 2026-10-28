/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "alpaka/onHost/Handle.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>

using namespace alpaka;
using namespace alpaka::onHost;

struct Count
{
    Count(uint32_t value) : count(value)
    {
    }

    uint32_t count = 0u;
};

using CountHandle = Handle<Count>;

TEST_CASE("singelton handle", "")
{
    {
        auto singleton = makeSharedSingleton<Count>(42);
        CHECK(singleton->count == 42);
        /* We have already a valid handle therefore the old should be returned.
         * This handle is only an alias.
         */
        auto singleton2 = makeSharedSingleton<Count>(43);
        CHECK(singleton->count == 42);
        singleton2->count = 1;
        CHECK(singleton->count == 1);
    }
    // old handles should be deleted
    auto singleton = makeSharedSingleton<Count>(23);
    CHECK(singleton->count == 23);
}
