/* Copyright 2024 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>

#include <iostream>
#include <type_traits>

using Dim = alpaka::DimInt<1>;
using Idx = std::size_t;

template<typename TTag>
constexpr bool isCPUTag()
{
    if constexpr(
        std::is_same_v<TTag, alpaka::TagCpuSerial> || std::is_same_v<TTag, alpaka::TagCpuThreads>
        || std::is_same_v<TTag, alpaka::TagCpuTbbBlocks> || std::is_same_v<TTag, alpaka::TagCpuOmp2Blocks>
        || std::is_same_v<TTag, alpaka::TagCpuOmp2Threads>)
    {
        return true;
    }
    else
    {
        return false;
    }
}

using TagList = std::tuple<
    alpaka::TagCpuSerial,
    alpaka::TagCpuThreads,
    alpaka::TagCpuTbbBlocks,
    alpaka::TagCpuOmp2Blocks,
    alpaka::TagCpuOmp2Threads,
    alpaka::TagGpuCudaRt,
    alpaka::TagGpuHipRt,
    alpaka::TagCpuSycl,
    alpaka::TagFpgaSyclIntel,
    alpaka::TagGpuSyclIntel>;

TEMPLATE_LIST_TEST_CASE("memoryVisibilityType", "[mem][visibility]", TagList)
{
    using Tag = TestType;
    if constexpr(alpaka::AccIsEnabled<Tag>::value)
    {
        using DevType = decltype(alpaka::getDevByIdx(alpaka::Platform<alpaka::TagToAcc<Tag, Dim, Idx>>{}, 0));
        if constexpr(isCPUTag<Tag>())
        {
            STATIC_REQUIRE(
                std::is_same_v<typename alpaka::trait::MemVisibility<DevType>::type, alpaka::MemVisibleCPU>);
        }
        else if(std::is_same_v<Tag, alpaka::TagGpuCudaRt>)
        {
            STATIC_REQUIRE(
                std::is_same_v<typename alpaka::trait::MemVisibility<DevType>::type, alpaka::MemVisibleGpuCudaRt>);
        }
    }
}

using TagTagList = alpaka::meta::CartesianProduct<std::tuple, TagList, TagList>;

template<typename TAcc, typename TDev, typename TBuf>
void do_job(TDev dev, TBuf buffer)
{
    STATIC_REQUIRE(alpaka::hasSameMemView(dev, buffer));
}

TEMPLATE_LIST_TEST_CASE("printDefines", "[mem][visibility]", TagTagList)
{
    using Tag1 = std::tuple_element_t<0, TestType>;
    using Tag2 = std::tuple_element_t<1, TestType>;

    if constexpr(alpaka::AccIsEnabled<Tag1>::value && alpaka::AccIsEnabled<Tag2>::value)
    {
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

        STATIC_REQUIRE(alpaka::hasSameMemView(dev1, bufDev1));
        STATIC_REQUIRE(alpaka::hasSameMemView(dev2, bufDev2));

        // at the moment, only the cpu platform has different accelerator types
        // therefore all cpu accelerators can access the memory of other cpu accelerators
        // if the accelerator is not a cpu accelerator, both accelerators needs to be the
        // same to support access to the memory of each other
        if constexpr((isCPUTag<Tag1>() && isCPUTag<Tag2>()) || std::is_same_v<Tag1, Tag2>)
        {
            STATIC_REQUIRE(alpaka::hasSameMemView(dev1, bufDev2));
            STATIC_REQUIRE(alpaka::hasSameMemView(dev2, bufDev1));
        }
        else
        {
            STATIC_REQUIRE_FALSE(alpaka::hasSameMemView(dev1, bufDev2));
            STATIC_REQUIRE_FALSE(alpaka::hasSameMemView(dev2, bufDev1));
        }

        do_job<Acc1>(dev1, bufDev1);
        do_job<Acc1>(dev2, bufDev2);
        // do_job<Acc1>(dev1, bufDev2);

        // std::cout << std::boolalpha << "tag 1 is cpu: " << isCPUTag<Tag1>() << "\n";
        // std::cout << std::boolalpha << "tag 2 is cpu: " << isCPUTag<Tag2>() << "\n";
    }
}
