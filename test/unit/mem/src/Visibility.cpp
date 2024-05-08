/* Copyright 2024 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>

#include <iostream>
#include <type_traits>

using Dim = alpaka::DimInt<1>;
using Idx = std::size_t;

TEMPLATE_LIST_TEST_CASE("memoryVisibilityType", "[mem][visibility]", alpaka::EnabledAccTags)
{
    using Tag = TestType;


    REQUIRE(true);
    using PltfType = alpaka::Platform<alpaka::TagToAcc<Tag, Dim, Idx>>;

    if constexpr(alpaka::isCpuTag<Tag>::value)
    {
        REQUIRE(std::is_same_v<typename alpaka::trait::MemVisibility<PltfType>::type, alpaka::MemVisibleCPU>);
    }
    else if(std::is_same_v<Tag, alpaka::TagGpuCudaRt>)
    {
        REQUIRE(std::is_same_v<typename alpaka::trait::MemVisibility<PltfType>::type, alpaka::MemVisibleGpuCudaRt>);
    }
    else if(std::is_same_v<Tag, alpaka::TagGpuSyclIntel>)
    {
        REQUIRE(std::is_same_v<typename alpaka::trait::MemVisibility<PltfType>::type, alpaka::MemVisibleGpuSyclIntel>);
    }
    else if(std::is_same_v<Tag, alpaka::TagGpuHipRt>)
    {
        REQUIRE(std::is_same_v<typename alpaka::trait::MemVisibility<PltfType>::type, alpaka::MemVisibleGpuHipRt>);
    }
    else if(std::is_same_v<Tag, alpaka::TagGenericSycl>)
    {
        REQUIRE(std::is_same_v<typename alpaka::trait::MemVisibility<PltfType>::type, alpaka::MemVisibleGenericSycl>);
    }
    else if(std::is_same_v<Tag, alpaka::TagFpgaSyclIntel>)
    {
        REQUIRE(
            std::is_same_v<typename alpaka::trait::MemVisibility<PltfType>::type, alpaka::MemVisibleFpgaSyclIntel>);
    }
}

using EnabledTagTagList = alpaka::meta::CartesianProduct<std::tuple, alpaka::EnabledAccTags, alpaka::EnabledAccTags>;

TEMPLATE_LIST_TEST_CASE("testHasSameMemView", "[mem][visibility]", EnabledTagTagList)
{
    using Tag1 = std::tuple_element_t<0, TestType>;
    using Tag2 = std::tuple_element_t<1, TestType>;

    SUCCEED(Tag1::get_name() << " + " << Tag2::get_name());

    using Acc1 = alpaka::TagToAcc<Tag1, Dim, Idx>;
    using Acc2 = alpaka::TagToAcc<Tag2, Dim, Idx>;

    auto const plt1 = alpaka::Platform<Acc1>{};
    auto const plt2 = alpaka::Platform<Acc2>{};

    auto const dev1 = alpaka::getDevByIdx(plt1, 0);
    auto const dev2 = alpaka::getDevByIdx(plt2, 0);

    using BufAcc1 = alpaka::Buf<Acc1, float, Dim, Idx>;
    using BufAcc2 = alpaka::Buf<Acc2, float, Dim, Idx>;

    BufAcc1 bufDev1(alpaka::allocBuf<float, Idx>(dev1, Idx(1)));
    BufAcc2 bufDev2(alpaka::allocBuf<float, Idx>(dev2, Idx(1)));

    STATIC_REQUIRE(alpaka::hasSameMemView(plt1, bufDev1));
    STATIC_REQUIRE(alpaka::hasSameMemView(dev1, bufDev1));
    STATIC_REQUIRE(alpaka::hasSameMemView<Acc1, BufAcc1>());
    STATIC_REQUIRE(alpaka::hasSameMemView(plt2, bufDev2));
    STATIC_REQUIRE(alpaka::hasSameMemView(dev2, bufDev2));
    STATIC_REQUIRE(alpaka::hasSameMemView<Acc2, BufAcc2>());

    // at the moment, only the cpu platform has different accelerator types
    // therefore all cpu accelerators can access the memory of other cpu accelerators
    // if the accelerator is not a cpu accelerator, both accelerators needs to be the
    // same to support access to the memory of each other
    if constexpr((alpaka::isCpuTag<Tag1>::value && alpaka::isCpuTag<Tag2>::value) || std::is_same_v<Tag1, Tag2>)
    {
        STATIC_REQUIRE(alpaka::hasSameMemView(plt1, bufDev2));
        STATIC_REQUIRE(alpaka::hasSameMemView(plt2, bufDev1));
        STATIC_REQUIRE(alpaka::hasSameMemView<Acc2, BufAcc1>());
    }
    else
    {
        STATIC_REQUIRE_FALSE(alpaka::hasSameMemView(plt1, bufDev2));
        STATIC_REQUIRE_FALSE(alpaka::hasSameMemView(plt2, bufDev1));
        STATIC_REQUIRE_FALSE(alpaka::hasSameMemView<Acc2, BufAcc1>());
    }
}
