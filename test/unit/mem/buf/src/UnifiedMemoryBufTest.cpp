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

// namespace managed_buftest
// {
// #if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
//     template<typename TElem, typename TIdx, typename TExtent, typename TPlatform>
//     auto allocManagedBuf(alpaka::DevCpu const& host, TPlatform const& platform, TExtent const& extent) 
//     {
//         return alpaka::allocManagedBuf<TElem, TIdx, TExtent, TPlatform>(host, platform, extent);
//     }
// #endif
// #if defined(ALPAKA_ACC_GPU_HIP_ENABLED)
//     template<typename TElem, typename TIdx, typename TExtent, typename TPlatform>
//     auto allocManagedBuf(alpaka::DevCpu const& host, TPlatform const& platform, TExtent const& extent) 
//     {
//         return alpaka::allocManagedBuf<TElem, TIdx, TExtent, TPlatform>(host, platform, extent);
//     }
// #endif
// } // namespace managed_buftest

// template<typename TAcc>
// static auto testManagedBuffer(alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>> const& extent) -> void
// {
//     using Dev = alpaka::Dev<TAcc>;
//     using Queue = alpaka::test::DefaultQueue<Dev>;
    
//     using Elem = float;
//     using Dim = alpaka::Dim<TAcc>;
//     using Idx = alpaka::Idx<TAcc>;
//     using Extent = alpaka::Vec<Dim, Idx>;
//     using Platform = alpaka::Platform<TAcc>;

//     auto const platformAcc = alpaka::Platform<TAcc>{};
//     auto const dev = alpaka::getDevByIdx(platformAcc, 0);
//     Queue queue(dev);

//     // alpaka::mallocManaged
//     //auto buf = alpaka::allocManagedBuf<Elem, Idx, Extent, Platform>(dev, platformAcc, extent);

//     auto const host = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0);
//     auto buf = alpaka::allocManagedBuf<Elem, Idx, Extent, Platform>(host, platformAcc, extent);

//     auto const offset = alpaka::Vec<Dim, Idx>::zeros();
//     alpaka::test::testViewImmutable<Elem>(buf, dev, extent, offset);

//     alpaka::test::testViewMutable<TAcc>(queue, buf);
// }


// TEMPLATE_LIST_TEST_CASE("memBufManagedTest", "[memBuf]", alpaka::test::TestAccs)
// {
//     using Acc = TestType;
//     using Dim = alpaka::Dim<Acc>;
//     using Idx = alpaka::Idx<Acc>;
//     testManagedBuffer<Acc>(alpaka::test::extentBuf<Dim, Idx>);
// }

// TEMPLATE_LIST_TEST_CASE("memBufManagedZeroSizeTest", "[memBuf]", alpaka::test::TestAccs)
// {
//     using Acc = TestType;
//     using Dim = alpaka::Dim<Acc>;
//     using Idx = alpaka::Idx<Acc>;

//     auto const extent = alpaka::Vec<Dim, Idx>::zeros();

//     testManagedBuffer<Acc>(extent);
// }



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

    auto buf = alpaka::allocManagedBuf<Elem, Idx, alpaka::Vec<Dim, Idx>, Platform>(devHost, platformAcc, extent);

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
