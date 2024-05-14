/* Copyright 2024 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>

#include <iostream>
#include <type_traits>

using Dim = alpaka::DimInt<1>;
using Idx = std::size_t;

// TODO(SimeonEhrig): Replace implementation. Instead using a list, specialize `alpaka::Platform` for
// tags to get the Memory Visiblity

//! \brief check if the accelerator related to the tag is bounded to the cpu platform
//! \tparam TTag alpaka tag type
template<typename TTag, typename = void>
struct isCpuTag : std::false_type
{
};

template<typename TTag>
struct isCpuTag<
    TTag,
    std::enable_if_t<
        // TAGCpuSycl is not included because it has it's own platform
        std::is_same_v<TTag, alpaka::TagCpuSerial> || std::is_same_v<TTag, alpaka::TagCpuThreads>
        || std::is_same_v<TTag, alpaka::TagCpuTbbBlocks> || std::is_same_v<TTag, alpaka::TagCpuOmp2Blocks>
        || std::is_same_v<TTag, alpaka::TagCpuOmp2Threads>>> : std::true_type
{
};

template<typename TTagMemView, typename = void>
struct AccIsEnabledMemVisibilities : std::false_type
{
};

template<typename TTagMemView>
struct AccIsEnabledMemVisibilities<
    TTagMemView,
    std::void_t<alpaka::TagToAcc<std::tuple_element_t<0, TTagMemView>, alpaka::DimInt<1>, int>>> : std::true_type
{
};

using ExpectedTagsMemVisibilities = alpaka::meta::Filter<
    std::tuple<
        std::tuple<alpaka::TagCpuSerial, alpaka::MemVisibleCPU>,
        std::tuple<alpaka::TagCpuThreads, alpaka::MemVisibleCPU>,
        std::tuple<alpaka::TagCpuTbbBlocks, alpaka::MemVisibleCPU>,
        std::tuple<alpaka::TagCpuOmp2Blocks, alpaka::MemVisibleCPU>,
        std::tuple<alpaka::TagCpuOmp2Threads, alpaka::MemVisibleCPU>,
        std::tuple<alpaka::TagGpuCudaRt, alpaka::MemVisibleGpuCudaRt>,
        std::tuple<alpaka::TagGpuHipRt, alpaka::MemVisibleGpuHipRt>,
        std::tuple<alpaka::TagCpuSycl, alpaka::MemVisibleCPU>,
        std::tuple<alpaka::TagFpgaSyclIntel, alpaka::MemVisibleFpgaSyclIntel>,
        std::tuple<alpaka::TagGpuSyclIntel, alpaka::MemVisibleGpuSyclIntel>>,
    AccIsEnabledMemVisibilities>;

TEMPLATE_LIST_TEST_CASE("memoryVisibilityType", "[mem][visibility]", ExpectedTagsMemVisibilities)
{
    using Tag = std::tuple_element_t<0, TestType>;
    using ExpectedMemVisibility = std::tuple_element_t<1, TestType>;

    using PltfType = alpaka::Platform<alpaka::TagToAcc<Tag, Dim, Idx>>;
    STATIC_REQUIRE(std::is_same_v<typename alpaka::trait::MemVisibility<PltfType>::type, ExpectedMemVisibility>);
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
    if constexpr((isCpuTag<Tag1>::value && isCpuTag<Tag2>::value) || std::is_same_v<Tag1, Tag2>)
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

using EnabledTagTagMemVisibilityList
    = alpaka::meta::CartesianProduct<std::tuple, ExpectedTagsMemVisibilities, ExpectedTagsMemVisibilities>;

TEMPLATE_LIST_TEST_CASE("testMemView", "[mem][visibility]", EnabledTagTagMemVisibilityList)
{
    using Tag1 = std::tuple_element_t<0, std::tuple_element_t<0, TestType>>;
    using ExpectedMemVisibilityForTag1 = std::tuple_element_t<1, std::tuple_element_t<0, TestType>>;
    using Tag2 = std::tuple_element_t<0, std::tuple_element_t<1, TestType>>;
    using ExpectedMemVisibilityForTag2 = std::tuple_element_t<1, std::tuple_element_t<1, TestType>>;

    SUCCEED(
        "Tag1: " << Tag1::get_name() << " + " << ExpectedMemVisibilityForTag1::get_name()
                 << "\nTag2: " << Tag2::get_name() << " + " << ExpectedMemVisibilityForTag1::get_name());


    constexpr Idx data_size = 10;

    using Acc1 = alpaka::TagToAcc<Tag1, Dim, Idx>;
    using Acc2 = alpaka::TagToAcc<Tag2, Dim, Idx>;

    auto const plt1 = alpaka::Platform<Acc1>{};
    auto const plt2 = alpaka::Platform<Acc2>{};

    auto const dev1 = alpaka::getDevByIdx(plt1, 0);
    auto const dev2 = alpaka::getDevByIdx(plt2, 0);

    using Vec1D = alpaka::Vec<alpaka::DimInt<1>, Idx>;
    Vec1D const extents(Vec1D::all(data_size));

    std::array<int, data_size> data;

    auto data_view1 = alpaka::createView(dev1, data.data(), extents);
    STATIC_REQUIRE(std::is_same_v<
                   typename alpaka::trait::MemVisibility<decltype(data_view1)>::type,
                   std::tuple<ExpectedMemVisibilityForTag1>>);
    STATIC_REQUIRE(alpaka::hasSameMemView(plt1, data_view1));

    auto data_view2 = alpaka::createView(dev2, data.data(), extents);
    STATIC_REQUIRE(std::is_same_v<
                   typename alpaka::trait::MemVisibility<decltype(data_view2)>::type,
                   std::tuple<ExpectedMemVisibilityForTag2>>);
    STATIC_REQUIRE(alpaka::hasSameMemView(plt2, data_view2));

    if constexpr(std::is_same_v<ExpectedMemVisibilityForTag1, ExpectedMemVisibilityForTag2>)
    {
        STATIC_REQUIRE(alpaka::hasSameMemView(plt1, data_view2));
        STATIC_REQUIRE(alpaka::hasSameMemView(plt2, data_view1));
    }
    else
    {
        STATIC_REQUIRE_FALSE(alpaka::hasSameMemView(plt1, data_view2));
        STATIC_REQUIRE_FALSE(alpaka::hasSameMemView(plt2, data_view1));
    }
}
