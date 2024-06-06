/* Copyright 2022 Sergei Bastrakov, Jan Stephan, Bernhard Manfred Gruber, Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

#include "FooVec.hpp"

#include <alpaka/acc/AccDevProps.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/workdiv/WorkDivHelpers.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

struct TestKernelWithManyRegisters
{
    template<typename TAcc>
    [[maybe_unused]] ALPAKA_FN_ACC auto operator()(TAcc const&, std::size_t val) const -> void
    {
        double var0 = 1.0;
        double var1 = 2.0;
        double var2 = 3.0;

        // Define many variables and use some calculations in order to prevent compiler optimization and make the
        // kernel use many registers (around 80 on sm_52). Using many registers per SM decreases the max number of
        // threads per block while this kernel is being run.

        // TODO: Use function templates to parametrize and shorten the code!
        double var3 = var2 + fmod(var2, 5);

        double var4 = var3 + fmod(var3, 5);
        double var5 = var4 + fmod(var4, 5);
        double var6 = var5 + fmod(var5, 5);
        double var7 = var6 + fmod(var6, 5);
        double var8 = var7 + fmod(var7, 5);
        double var9 = var8 + fmod(var8, 5);
        double var10 = var9 + fmod(var9, 5);
        double var11 = var10 + fmod(var10, 5);
        double var12 = var11 + fmod(var11, 5);
        double var13 = var12 + fmod(var12, 5);
        double var14 = var13 + fmod(var13, 5);
        double var15 = var14 + fmod(var14, 5);
        double var16 = var15 + fmod(var15, 5);
        double var17 = var16 + fmod(var16, 5);
        double var18 = var17 + fmod(var17, 5);
        double var19 = var18 + fmod(var18, 5);
        double var20 = var19 + fmod(var19, 5);
        double var21 = var20 + fmod(var20, 5);
        double var22 = var21 + fmod(var21, 5);
        double var23 = var22 + fmod(var22, 5);
        double var24 = var23 + fmod(var23, 5);
        double var25 = var24 + fmod(var24, 5);
        double var26 = var25 + fmod(var25, 5);
        double var27 = var26 + fmod(var26, 5);
        double var28 = var27 + fmod(var27, 5);
        double var29 = var28 + fmod(var28, 5);
        double var30 = var29 + fmod(var29, 5);
        double var31 = var30 + fmod(var30, 5);
        double var32 = var31 + fmod(var31, 5);
        double var33 = var32 + fmod(var32, 5);
        double var34 = var33 + fmod(var33, 5);
        double var35 = var34 + fmod(var34, 5);

        double sum = var0 + var1 + var2 + var3 + var4 + var5 + var6 + var7 + var8 + var9 + var10 + var11 + var12
                     + var13 + var14 + var15 + var16 + var17 + var18 + var19 + var20 + var21 + var22 + var23 + var24
                     + var25 + var26 + var27 + var28 + var29 + var30 + var31 + var32 + var33 + var34 + var35;
        printf("The sum is %5.2f, the argument is %lu ", sum, val);
    }
};

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
//! Test using any vector type in WorkDivMembers construction.
TEST_CASE("WorkDivMembers", "[workDiv]")
{
    using Idx = std::size_t;
    using Dim3D = alpaka::DimInt<3>;
    using Vec3D = alpaka::Vec<Dim3D, Idx>;

    auto blocksPerGrid3D = Vec3D{1u, 2u, 3u};
    auto const threadsPerBlock3D = Vec3D{2u, 4u, 6u};
    auto const elementsPerThread3D = Vec3D::all(static_cast<Idx>(1u));

    // Arguments: {1,2,3},{2,4,6},{1,1,1}
    auto ref3D = alpaka::WorkDivMembers<Dim3D, Idx>{blocksPerGrid3D, threadsPerBlock3D, elementsPerThread3D};
    // Call without explicitly specifying explicit WorkDivMembers class template parameter types
    auto workDiv3D = alpaka::WorkDivMembers(blocksPerGrid3D, threadsPerBlock3D, elementsPerThread3D);
    CHECK(ref3D == workDiv3D);

    // Change blocks per grid
    blocksPerGrid3D = Vec3D{3u, 6u, 9u};
    // Arguments: {3,6,9},{2,4,6},{1,1,1}
    ref3D = alpaka::WorkDivMembers<Dim3D, Idx>{blocksPerGrid3D, threadsPerBlock3D, elementsPerThread3D};
    // Call without explicitly specifying explicit WorkDivMembers class template parameter types
    workDiv3D = alpaka::WorkDivMembers(blocksPerGrid3D, threadsPerBlock3D, elementsPerThread3D);
    CHECK(ref3D == workDiv3D);

    // Test using 2D vectors
    using Dim2D = alpaka::DimInt<2>;
    using Vec2D = alpaka::Vec<Dim2D, Idx>;

    auto const blocksPerGrid2D = Vec2D{6u, 9u};
    auto const threadsPerBlock2D = Vec2D{4u, 6u};
    auto const elementsPerThread2D = Vec2D::all(static_cast<Idx>(1u));

    // Arguments: {6,9},{4,6},{1,1}. The order of each input is y-x since alpaka vector uses z-y-x ordering
    auto const ref2D = alpaka::WorkDivMembers<Dim2D, Idx>{blocksPerGrid2D, threadsPerBlock2D, elementsPerThread2D};
    auto const workDiv2D = alpaka::WorkDivMembers(blocksPerGrid2D, threadsPerBlock2D, elementsPerThread2D);
    CHECK(ref2D == workDiv2D);

    // Test using initializer lists. Arguments: {6,9},{4,6},{1,1}. The order of initializer list is y-x since alpaka
    // vector uses z-y-x ordering
    auto const workDiv2DUsingInitList = alpaka::WorkDivMembers<Dim2D, Idx>({6, 9}, {4, 6}, {1, 1});
    CHECK(ref2D == workDiv2DUsingInitList);

    // Test using different input types with different number of dimensions(ranks), number of dimensions reduced to
    // given explicit class template type number of dimensions (e.g. Dim2D) in the call. Arguments: {6,9},{2,4,6},{1,1}
    auto worDiv2DUsingMixedRanks
        = alpaka::WorkDivMembers<Dim2D, Idx>{blocksPerGrid2D, threadsPerBlock3D, elementsPerThread3D};
    // Since the first element of threadsPerBlock3D is along Z-axis, it is removed
    CHECK(ref2D == worDiv2DUsingMixedRanks);

    worDiv2DUsingMixedRanks
        = alpaka::WorkDivMembers<Dim2D, Idx>{blocksPerGrid2D, threadsPerBlock3D, elementsPerThread2D};
    CHECK(ref2D == worDiv2DUsingMixedRanks);

    // Test the construction by using a user-defined type FooVec
    //
    // Test WorkDivMembers using the arguments of the type FooVec
    auto const blocksPerGrid2DFoo = FooVec<size_t, 2u>{9u, 6u};
    auto const threadsPerBlock2DFoo = FooVec<size_t, 2u>{6u, 4u};
    auto const elementsPerThread2DFoo = FooVec<size_t, 2u>{1u, 1u};

    // Arguments: {9,6},{6,4},{1,1}. These arguments are reversed at GetExtents specialization of FooVec
    // FooVec assumes the list is ordered as x-y-z
    auto const workDiv2DUsingFooVec
        = alpaka::WorkDivMembers<Dim2D, Idx>{blocksPerGrid2DFoo, threadsPerBlock2DFoo, elementsPerThread2DFoo};
    CHECK(ref2D == workDiv2DUsingFooVec);
}

using TestAccs1D = alpaka::test::EnabledAccs<alpaka::DimInt<1u>, std::uint32_t>;

TEMPLATE_LIST_TEST_CASE("testVectorCreation", "[workDiv]", TestAccs1D)
{
    using Acc = TestType;
    using Idx = alpaka::Idx<Acc>;
    printf("%u", static_cast<Idx>(1u));

    // using Dim = alpaka::Dim<Acc>;
    // using Vec = alpaka::Vec<Dim, Idx>;
    using Dim = alpaka::Dim<Acc>;
    using Vec = alpaka::Vec<Dim, Idx>;


    // Vec const gridThreadExtent(static_cast<Idx>(1), static_cast<Idx>(threadsPerGridTestValue / 8));
    Vec const gridThreadExtent(static_cast<Idx>(1));
    printf("%u", gridThreadExtent[0]);
}

using TestAccs2D = alpaka::test::EnabledAccs<alpaka::DimInt<2u>, std::uint32_t>;

TEMPLATE_LIST_TEST_CASE("testVectorCreation", "[workDiv]", TestAccs2D)
{
    using Acc = TestType;
    using Idx = alpaka::Idx<Acc>;
    printf("%u", static_cast<Idx>(2u));

    // using Dim = alpaka::Dim<Acc>;
    // using Vec = alpaka::Vec<Dim, Idx>;
    using Dim = alpaka::Dim<Acc>;
    using Vec = alpaka::Vec<Dim, Idx>;
    // using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;
    auto const platform = alpaka::Platform<Acc>{};
    auto const dev = alpaka::getDevByIdx(platform, 0);

    // Vec const gridThreadExtent(static_cast<Idx>(1), static_cast<Idx>(threadsPerGridTestValue / 8));
    Vec const gridThreadExtent(static_cast<Idx>(1), static_cast<Idx>(800 / 8));
    printf("%u", gridThreadExtent[0]);
    auto blocksPerGrid3D = Vec{1u, 2u};


    TestKernelWithManyRegisters kernel;


    // Get hard limits for test
    // auto const& props = alpaka::getAccDevProps<Acc>(dev);
    // Idx const threadsPerGridTestValue = props.m_blockThreadCountMax * props.m_gridBlockCountMax;

    // Test getValidWorkDivForKernel for threadsPerGridTestValue threads per grid
    auto const workDiv = alpaka::getValidWorkDiv<Acc>(dev, gridThreadExtent, Vec{1, 1});
    // Test validity
    auto const isValid = alpaka::isValidWorkDiv<Acc>(dev, workDiv);
    CHECK(isValid == true);
}
