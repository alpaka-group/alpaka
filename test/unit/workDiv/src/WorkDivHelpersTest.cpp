/* Copyright 2022 Sergei Bastrakov, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/acc/AccDevProps.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/workdiv/WorkDivHelpers.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tuple>

namespace
{
    template<typename TAcc>
    auto getWorkDiv()
    {
        using Dim = alpaka::Dim<TAcc>;
        using Idx = alpaka::Idx<TAcc>;

        auto const platform = alpaka::Platform<TAcc>{};
        auto const dev = alpaka::getDevByIdx(platform, 0);
        auto const gridThreadExtent = alpaka::Vec<Dim, Idx>::all(10);
        auto const threadElementExtent = alpaka::Vec<Dim, Idx>::ones();
        auto const workDiv = alpaka::getValidWorkDiv<TAcc>(
            dev,
            gridThreadExtent,
            threadElementExtent,
            false,
            alpaka::GridBlockExtentSubDivRestrictions::Unrestricted);
        return workDiv;
    }
} // namespace

TEMPLATE_LIST_TEST_CASE("getValidWorkDiv", "[workDiv]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    // Note: getValidWorkDiv() is called inside getWorkDiv
    std::ignore = getWorkDiv<Acc>();
}

TEMPLATE_LIST_TEST_CASE("subDivideGridElems.2D.examples", "[workDiv]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Idx = alpaka::Idx<Acc>;
    using Dim = alpaka::Dim<Acc>;
    using Vec = alpaka::Vec<Dim, Idx>;
    using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;
    if constexpr(Dim::value == 2)
    {
        auto const platform = alpaka::Platform<Acc>{};
        auto const dev = alpaka::getDevByIdx(platform, 0);
        auto props = alpaka::getAccDevProps<Acc>(dev);
        props.m_gridBlockExtentMax = Vec{1024, 1024};
        props.m_gridBlockCountMax = 1024 * 1024;
        props.m_blockThreadExtentMax = Vec{256, 128};
        props.m_blockThreadCountMax = 512;
        props.m_threadElemExtentMax = Vec{8, 8};
        props.m_threadElemCountMax = 16;

        CHECK(
            alpaka::subDivideGridElems(
                Vec{300, 600},
                Vec{1, 1},
                props,
                false,
                alpaka::GridBlockExtentSubDivRestrictions::EqualExtent)
            == WorkDiv{Vec{14, 28}, Vec{22, 22}, Vec{1, 1}});
        CHECK(
            alpaka::subDivideGridElems(
                Vec{300, 600},
                Vec{1, 1},
                props,
                false,
                alpaka::GridBlockExtentSubDivRestrictions::CloseToEqualExtent)
            == WorkDiv{Vec{19, 19}, Vec{16, 32}, Vec{1, 1}});
        CHECK(
            alpaka::subDivideGridElems(
                Vec{300, 600},
                Vec{1, 1},
                props,
                false,
                alpaka::GridBlockExtentSubDivRestrictions::Unrestricted)
            == WorkDiv{Vec{75, 5}, Vec{4, 128}, Vec{1, 1}});

        CHECK(
            alpaka::subDivideGridElems(
                Vec{300, 600},
                Vec{1, 1},
                props,
                true,
                alpaka::GridBlockExtentSubDivRestrictions::EqualExtent)
            == WorkDiv{Vec{1, 2}, Vec{300, 300}, Vec{1, 1}});
        CHECK(
            alpaka::subDivideGridElems(
                Vec{300, 600},
                Vec{1, 1},
                props,
                true,
                alpaka::GridBlockExtentSubDivRestrictions::CloseToEqualExtent)
            == WorkDiv{Vec{20, 20}, Vec{15, 30}, Vec{1, 1}});
        CHECK(
            alpaka::subDivideGridElems(
                Vec{300, 600},
                Vec{1, 1},
                props,
                true,
                alpaka::GridBlockExtentSubDivRestrictions::Unrestricted)
            == WorkDiv{Vec{75, 5}, Vec{4, 120}, Vec{1, 1}});
    }
}

TEMPLATE_LIST_TEST_CASE("getValidWorkDiv.1D.withIdx", "[workDiv]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Idx = alpaka::Idx<Acc>;
    using Dim = alpaka::Dim<Acc>;
    using Vec = alpaka::Vec<Dim, Idx>;
    if constexpr(Dim::value == 1)
    {
        auto const platform = alpaka::Platform<Acc>{};
        auto const dev = alpaka::getDevByIdx(platform, 0);
        // test that we can call getValidWorkDiv with the Idx type directly instead of a Vec
        auto const ref = alpaka::getValidWorkDiv<Acc>(dev, Vec{256}, Vec{13});
        CHECK(alpaka::getValidWorkDiv<Acc>(dev, Idx{256}, Idx{13}) == ref);
    }
}

TEMPLATE_LIST_TEST_CASE("isValidWorkDiv", "[workDiv]", alpaka::test::TestAccs)
{
    using Acc = TestType;

    auto const platform = alpaka::Platform<Acc>{};
    auto const dev = alpaka::getDevByIdx(platform, 0);
    auto const workDiv = getWorkDiv<Acc>();
    // Test both overloads
    REQUIRE(alpaka::isValidWorkDiv(alpaka::getAccDevProps<Acc>(dev), workDiv));
    REQUIRE(alpaka::isValidWorkDiv<Acc>(dev, workDiv));
}

//! Test the constructors of WorkDivMembers using 3D extent, 3D extent with zero elements and 2D extents
TEST_CASE("WorkDivMembers", "[workDiv]")
{
    using Idx = std::size_t;
    using Dim3D = alpaka::DimInt<3>;
    using Vec3D = alpaka::Vec<Dim3D, Idx>;

    auto const elementsPerThread3D = Vec3D::all(static_cast<Idx>(1u));
    auto const threadsPerBlock3D = Vec3D{2u, 2u, 2u};
    auto blocksPerGrid3D = Vec3D{1u, 1u, 1u};

    auto ref3D = alpaka::WorkDivMembers<Dim3D, Idx>{blocksPerGrid3D, threadsPerBlock3D, elementsPerThread3D};
    // call WorkDivMembers without explicit class template types
    auto workDiv3D = alpaka::WorkDivMembers(blocksPerGrid3D, threadsPerBlock3D, elementsPerThread3D);
    CHECK(workDiv3D == ref3D);

    // change blocks per grid, assign zero to an element
    blocksPerGrid3D = Vec3D{3u, 3u, 0u};
    ref3D = alpaka::WorkDivMembers<Dim3D, Idx>{blocksPerGrid3D, threadsPerBlock3D, elementsPerThread3D};
    // call without explicit template parameter types
    workDiv3D = alpaka::WorkDivMembers(blocksPerGrid3D, threadsPerBlock3D, elementsPerThread3D);
    CHECK(workDiv3D == ref3D);

    // test using 2D vectors
    using Dim2D = alpaka::DimInt<2>;
    using Vec2D = alpaka::Vec<Dim2D, Idx>;

    auto const threadsPerBlock2D = Vec2D{2u, 2u};
    auto const blocksPerGrid2D = Vec2D{1u, 1u};
    auto const elementsPerThread2D = Vec2D::all(static_cast<Idx>(1u));
    auto const ref2D = alpaka::WorkDivMembers<Dim2D, Idx>{blocksPerGrid2D, threadsPerBlock2D, elementsPerThread2D};
    auto const workDiv2D = alpaka::WorkDivMembers(blocksPerGrid2D, threadsPerBlock2D, elementsPerThread2D);
    CHECK(workDiv2D == ref2D);

    // Test using different input types, reduced to given explicit class template types
    auto ref2DimUsingMixed
        = alpaka::WorkDivMembers<Dim2D, Idx>{blocksPerGrid2D, threadsPerBlock3D, elementsPerThread3D};
    CHECK(workDiv2D == ref2DimUsingMixed);

    ref2DimUsingMixed = alpaka::WorkDivMembers<Dim2D, Idx>{blocksPerGrid2D, threadsPerBlock3D, elementsPerThread2D};
    CHECK(workDiv2D == ref2DimUsingMixed);
}
