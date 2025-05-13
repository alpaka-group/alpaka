/* Copyright 2025 Anton Reinhard
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

namespace buftest
{
    template<typename TDim, typename TElem, typename TIdx, typename TExtent>
    auto allocConstBuf(alpaka::DevCpu dev, TExtent extent) -> alpaka::ConstBufCpu<TElem, TDim, TIdx>
    {
        return alpaka::ConstBufCpu<TElem, TDim, TIdx>(alpaka::allocBuf<TElem, TIdx>(dev, extent));
    }

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    template<typename TDim, typename TElem, typename TIdx, typename TExtent>
    auto allocConstBuf(alpaka::DevCudaRt dev, TExtent extent) -> alpaka::ConstBufCudaRt<TElem, TDim, TIdx>
    {
        return alpaka::ConstBufCudaRt<TElem, TDim, TIdx>(alpaka::allocBuf<TElem, TIdx>(dev, extent));
    }
#endif
#if defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    template<typename TDim, typename TElem, typename TIdx, typename TExtent>
    auto allocConstBuf(alpaka::DevHipRt dev, TExtent extent) -> alpaka::ConstBufHipRt<TElem, TDim, TIdx>
    {
        return alpaka::ConstBufHipRt<TElem, TDim, TIdx>(alpaka::allocBuf<TElem, TIdx>(dev, extent));
    }
#endif
#if defined(ALPAKA_ACC_SYCL_ENABLED) and defined(ALPAKA_SYCL_ONEAPI_CPU)
    template<typename TDim, typename TElem, typename TIdx, typename TExtent>
    auto allocConstBuf(alpaka::DevCpuSycl dev, TExtent extent) -> alpaka::ConstBufCpuSycl<TElem, TDim, TIdx>
    {
        return alpaka::ConstBufCpuSycl<TElem, TDim, TIdx>(alpaka::allocBuf<TElem, TIdx>(dev, extent));
    }
#endif
#if defined(ALPAKA_ACC_SYCL_ENABLED) and defined(ALPAKA_SYCL_ONEAPI_GPU)
    template<typename TDim, typename TElem, typename TIdx, typename TExtent>
    auto allocConstBuf(alpaka::DevGpuSyclIntel dev, TExtent extent) -> alpaka::ConstBufGpuSyclIntel<TElem, TDim, TIdx>
    {
        return alpaka::ConstBufGpuSyclIntel<TElem, TDim, TIdx>(alpaka::allocBuf<TElem, TIdx>(dev, extent));
    }
#endif
#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_FPGA)
    template<typename TDim, typename TElem, typename TIdx, typename TExtent>
    auto allocConstBuf(alpaka::DevFpgaSyclIntel dev, TExtent extent)
        -> alpaka::ConstBufFpgaSyclIntel<TElem, TDim, TIdx>
    {
        return alpaka::ConstBufFpgaSyclIntel<TElem, TDim, TIdx>(alpaka::allocBuf<TElem, TIdx>(dev, extent));
    }
#endif

    template<typename TBuf>
    concept onlyConstNativePtr = requires(TBuf t) {
        { alpaka::getPtrNative(t) } -> std::same_as<alpaka::Elem<TBuf> const*>;
    };

} // namespace buftest

template<typename TAcc>
static auto testConstBuffer(alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>> const& extent) -> void
{
    using Dev = alpaka::Dev<TAcc>;
    using Queue = alpaka::test::DefaultQueue<Dev>;

    using Elem = float;
    using Dim = alpaka::Dim<TAcc>;
    using Idx = alpaka::Idx<TAcc>;

    auto const platformAcc = alpaka::Platform<TAcc>{};
    auto const dev = alpaka::getDevByIdx(platformAcc, 0);
    Queue queue(dev);

    // alpaka::malloc
    auto buf = alpaka::allocBuf<Elem, Idx>(dev, extent);
    auto const c_buf = buftest::allocConstBuf<Dim, Elem, Idx>(dev, extent);

    using TBuf = decltype(buf);
    using TCBuf = decltype(c_buf);

    auto const offset = alpaka::Vec<Dim, Idx>::zeros();
    alpaka::test::testViewImmutable<Elem const>(c_buf, dev, extent, offset);

    // check that Constant buffer can't be converted to non-const buffer
    STATIC_REQUIRE_FALSE(std::convertible_to<TCBuf, TBuf>);
    STATIC_REQUIRE(std::convertible_to<TCBuf, TCBuf>);
    STATIC_REQUIRE(std::convertible_to<TBuf, TBuf>);
    STATIC_REQUIRE(std::convertible_to<TBuf, TCBuf>);

    STATIC_REQUIRE_FALSE(buftest::onlyConstNativePtr<TBuf>);
}

TEMPLATE_LIST_TEST_CASE("constMemBufBasicTest", "[memBuf]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;
    testConstBuffer<Acc>(alpaka::test::extentBuf<Dim, Idx>);
}
