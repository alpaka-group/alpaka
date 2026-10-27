/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: Apache-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("IdxRange::begin() and end()", "[mem][IdxRange][iterator]")
{
    SECTION("only end extents")
    {
        constexpr int y = 4;
        for(int counter = 0; auto vec : alpaka::IdxRange(alpaka::Vec{y, 3}))
        {
            REQUIRE(vec == alpaka::Vec{counter / (y - 1), counter % (y - 1)});
            counter++;
        }
    }

    SECTION("begin and end extents")
    {
        constexpr int y = 4;
        constexpr int x = 3;
        constexpr int yBegin = 2;
        constexpr int xBegin = 1;
        for(auto expectedVec = alpaka::Vec{yBegin, xBegin};
            auto vec : alpaka::IdxRange(alpaka::Vec{yBegin, xBegin}, alpaka::Vec{y, x}))
        {
            REQUIRE(vec == expectedVec);
            expectedVec.x()++;
            if(expectedVec.x() == x)
            {
                expectedVec.y()++;
                expectedVec.x() = xBegin;
            }
        }
    }

    SECTION("begin and end extents and stride")
    {
        constexpr int y = 15;
        constexpr int x = 12;
        constexpr int yBegin = 2;
        constexpr int xBegin = 0;
        constexpr int yStride = 4;
        constexpr int xStride = 3;
        for(auto expectedVec = alpaka::Vec{yBegin, xBegin};
            auto vec : alpaka::IdxRange(alpaka::Vec{yBegin, xBegin}, alpaka::Vec{y, x}, alpaka::Vec{yStride, xStride}))
        {
            REQUIRE(vec == expectedVec);
            expectedVec.x() += xStride;
            if(expectedVec.x() >= x)
            {
                expectedVec.y() += yStride;
                expectedVec.x() = xBegin;
            }
        }
    }
}
