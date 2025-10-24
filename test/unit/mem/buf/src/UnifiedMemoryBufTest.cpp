/* Copyright 2025 Maria Michailidi
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/test/Extent.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/mem/view/ViewTest.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch_message.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <numeric>
#include <type_traits>

// kernel that changes the elements of a buffer
struct Kernel
{
    template <typename Acc, typename Buf>
    ALPAKA_FN_ACC void operator()(Acc const& acc, Buf buf, int value) const
    {
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        auto const extent = alpaka::getExtentProduct(buf);

        if(idx < extent)
        {
            auto* data = alpaka::getPtrNative(buf);
            data[idx] += value;
        }
    }
};

TEMPLATE_LIST_TEST_CASE("memBufManagedTest", "[memBuf]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dev = alpaka::Dev<Acc>;
    using Queue = alpaka::test::DefaultQueue<Dev>;
    using Elem = int;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;
    using Platform = alpaka::Platform<Acc>;

    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);

    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const dev = alpaka::getDevByIdx(platformAcc, 0);

    INFO("Test if unified memory works in: ");
    INFO(alpaka::getName(dev));

    Queue queue(dev);

    auto const extent = alpaka::test::extentBuf<Dim, Idx>;

    auto buf = alpaka::allocManagedBuf<Elem, Idx>(devHost, platformAcc, extent);

    constexpr Elem fillVal = 42;
    alpaka::fill(queue, buf, fillVal);

    alpaka::wait(queue);

    // constexpr int value = 10;

    // Idx const tpb = 256;
    // Idx const ept = 1;
    // Idx const blocks = (extent + tpb * ept - 1) / (tpb * ept);

    // using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;
    // auto div = WorkDiv{blocks, tpb, ept};

    // alpaka::exec<Acc>(
    //     queue,
    //     div,
    //     Kernel{},
    //     buf,
    //     value
    // );

    // alpaka::wait(queue);

    Idx const size = alpaka::getExtentProduct(buf);
    auto* data = alpaka::getPtrNative(buf); 
    bool passed = true;
    for(Idx i = 0; i < size; ++i)
    {
        if(data[i] != fillVal /*+ value*/)
        {
            passed = false;
        }
    }
    CHECK(passed);
}



TEMPLATE_LIST_TEST_CASE("memBufMappedTest", "[memBuf]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dev = alpaka::Dev<Acc>;
    using Queue = alpaka::test::DefaultQueue<Dev>;
    using Elem = int;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;
    using Platform = alpaka::Platform<Acc>;

    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);

    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const dev = alpaka::getDevByIdx(platformAcc, 0);

    INFO("Test if unified memory works in: ");
    INFO(alpaka::getName(dev));

    Queue queue(dev);

    auto const extent = alpaka::test::extentBuf<Dim, Idx>;

    auto buf = alpaka::allocMappedBuf<Elem, Idx, alpaka::Vec<Dim, Idx>, Platform>(devHost, platformAcc, extent);
    // maybe the last 2 template arguments are not necessary

    constexpr Elem fillVal = 42;
    alpaka::fill(queue, buf, fillVal);

    alpaka::wait(queue);

    Idx const size = alpaka::getExtentProduct(buf);
    auto* data = alpaka::getPtrNative(buf); 
    bool passed = true;
    for(Idx i = 0; i < size; ++i)
    {
        if(data[i] != fillVal)
        {
            passed = false;
        }
    }
    CHECK(passed);
}
