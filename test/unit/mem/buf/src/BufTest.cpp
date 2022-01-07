/* Copyright 2021 Axel Huebl, Benjamin Worpitz, Andrea Bocci
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/test/Extent.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/mem/view/ViewTest.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch.hpp>

#include <numeric>
#include <type_traits>

template<typename TAcc>
static constexpr auto isAsyncBufferSupported() -> bool
{
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    if constexpr(std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevCudaRt>)
    {
        return (BOOST_LANG_CUDA >= BOOST_VERSION_NUMBER(11, 2, 0)) && (alpaka::Dim<TAcc>::value == 1);
    }
    else
#endif // ALPAKA_ACC_GPU_CUDA_ENABLED

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
        if constexpr(std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevHipRt>)
    {
        return false;
    }
    else
#endif // ALPAKA_ACC_GPU_HIP_ENABLED

#ifdef ALPAKA_ACC_ANY_BT_OACC_ENABLED
        if constexpr(std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevOacc>)
    {
        return false;
    }
    else
#endif // ALPAKA_ACC_ANY_BT_OACC_ENABLED

#ifdef ALPAKA_ACC_ANY_BT_OMP5_ENABLED
        if constexpr(std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevOmp5>)
    {
        return false;
    }
    else
#endif // ALPAKA_ACC_ANY_BT_OMP5_ENABLED

        return true;

    ALPAKA_UNREACHABLE(bool{});
}

template<typename TAcc>
static auto testBufferMutable(alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>> const& extent) -> void
{
    using Dev = alpaka::Dev<TAcc>;
    using Pltf = alpaka::Pltf<Dev>;
    using Queue = alpaka::test::DefaultQueue<Dev>;

    using Elem = float;
    using Dim = alpaka::Dim<TAcc>;
    using Idx = alpaka::Idx<TAcc>;

    Dev const dev = alpaka::getDevByIdx<Pltf>(0u);
    Queue queue(dev);

    // alpaka::malloc
    auto buf = alpaka::allocBuf<Elem, Idx>(dev, extent);

    auto const offset = alpaka::Vec<Dim, Idx>::zeros();
    alpaka::test::testViewImmutable<Elem>(buf, dev, extent, offset);

    alpaka::test::testViewMutable<TAcc>(queue, buf);
}

template<typename TAcc>
static auto testAsyncBufferMutable(alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>> const& extent) -> void
{
    using Dev = alpaka::Dev<TAcc>;
    using Pltf = alpaka::Pltf<Dev>;
    using Queue = alpaka::test::DefaultQueue<Dev>;

    using Elem = float;
    using Dim = alpaka::Dim<TAcc>;
    using Idx = alpaka::Idx<TAcc>;

    Dev const dev = alpaka::getDevByIdx<Pltf>(0u);
    Queue queue(dev);

    // memory is allocated when the queue reaches this point
    auto buf = alpaka::allocAsyncBuf<Elem, Idx>(queue, extent);

    // asynchronous operations can be submitted to the queue immediately
    alpaka::test::testViewMutable<TAcc>(queue, buf);

    // synchronous operations must wait for the memory to be available
    alpaka::wait(queue);
    auto const offset = alpaka::Vec<Dim, Idx>::zeros();
    alpaka::test::testViewImmutable<Elem>(buf, dev, extent, offset);

    // the buffer will queue the deallocation of the memory when it goes out of scope,
    // and extend the lifetime of the queue until all memory operations have completed.
}

TEMPLATE_LIST_TEST_CASE("memBufBasicTest", "[memBuf]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    auto const extent
        = alpaka::createVecFromIndexedFn<Dim, alpaka::test::CreateVecWithIdx<Idx>::template ForExtentBuf>();

    testBufferMutable<Acc>(extent);
}

TEMPLATE_LIST_TEST_CASE("memBufZeroSizeTest", "[memBuf]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    auto const extent = alpaka::Vec<Dim, Idx>::zeros();

    testBufferMutable<Acc>(extent);
}

TEMPLATE_LIST_TEST_CASE("memBufAsyncBasicTest", "[memBuf]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    if constexpr(isAsyncBufferSupported<Acc>())
    {
        auto const extent
            = alpaka::createVecFromIndexedFn<Dim, alpaka::test::CreateVecWithIdx<Idx>::template ForExtentBuf>();
        testAsyncBufferMutable<Acc>(extent);
    }
    else
    {
        INFO("Stream-ordered memory buffers are not supported in this configuration.")
    }
}

TEMPLATE_LIST_TEST_CASE("memBufAsyncZeroSizeTest", "[memBuf]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    if constexpr(isAsyncBufferSupported<Acc>())
    {
        auto const extent = alpaka::Vec<Dim, Idx>::zeros();
        testAsyncBufferMutable<Acc>(extent);
    }
    else
    {
        INFO("Stream-ordered memory buffers are not supported in this configuration.")
    }
}

template<typename TAcc>
static auto testBufferImmutable(alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>> const& extent) -> void
{
    using Dev = alpaka::Dev<TAcc>;
    using Pltf = alpaka::Pltf<Dev>;

    using Elem = float;
    using Dim = alpaka::Dim<TAcc>;
    using Idx = alpaka::Idx<TAcc>;

    Dev const dev = alpaka::getDevByIdx<Pltf>(0u);

    // alpaka::malloc
    auto const buf = alpaka::allocBuf<Elem, Idx>(dev, extent);

    auto const offset = alpaka::Vec<Dim, Idx>::zeros();
    alpaka::test::testViewImmutable<Elem>(buf, dev, extent, offset);
}

TEMPLATE_LIST_TEST_CASE("memBufConstTest", "[memBuf]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    auto const extent
        = alpaka::createVecFromIndexedFn<Dim, alpaka::test::CreateVecWithIdx<Idx>::template ForExtentBuf>();

    testBufferImmutable<Acc>(extent);
}

template<typename TAcc>
static auto testAsyncBufferImmutable(alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>> const& extent) -> void
{
    using Dev = alpaka::Dev<TAcc>;
    using Pltf = alpaka::Pltf<Dev>;
    using Queue = alpaka::test::DefaultQueue<Dev>;

    using Elem = float;
    using Dim = alpaka::Dim<TAcc>;
    using Idx = alpaka::Idx<TAcc>;

    Dev const dev = alpaka::getDevByIdx<Pltf>(0u);
    Queue queue(dev);

    // memory is allocated when the queue reaches this point
    auto const buf = alpaka::allocAsyncBuf<Elem, Idx>(queue, extent);

    // synchronous operations must wait for the memory to be available
    alpaka::wait(queue);
    auto const offset = alpaka::Vec<Dim, Idx>::zeros();
    alpaka::test::testViewImmutable<Elem>(buf, dev, extent, offset);

    // the buffer will queue the deallocation of the memory when it goes out of scope,
    // and extend the lifetime of the queue until all memory operations have completed.
}

TEMPLATE_LIST_TEST_CASE("memBufAsyncConstTest", "[memBuf]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    if constexpr(isAsyncBufferSupported<Acc>())
    {
        auto const extent
            = alpaka::createVecFromIndexedFn<Dim, alpaka::test::CreateVecWithIdx<Idx>::template ForExtentBuf>();
        testAsyncBufferImmutable<Acc>(extent);
    }
    else
    {
        INFO("Stream-ordered memory buffers are not supported in this configuration.")
    }
}
