/* SPDX-License-Identifier: MPL-2.0
 */


#include <alpaka/alpaka.hpp>
#include <alpaka/test/Extent.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>
#include <alpaka/warp/Traits.hpp>

#include </afs/hep.wisc.edu/home/astrel/work/Alpaka/alpaka/include/alpaka/algo/WarpAlgos.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>

template<std::uint32_t TWarpSize>
struct WarpAlgosTestKernel
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        auto const localThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
        auto const blockExtent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        std::int32_t const warpExtent = alpaka::warp::getSize(acc);
        // Test relies on having a single warp per thread block
        ALPAKA_CHECK(*success, static_cast<std::int32_t>(blockExtent.prod()) == warpExtent);

        auto const threadIdxInWarp = std::uint32_t(alpaka::mapIdx<1u>(localThreadIdx, blockExtent)[0]);

        auto const mask = alpaka::warp::activemask(acc);
        using MaskType = decltype(mask);

        ALPAKA_CHECK(*success, warpExtent >= 1);

        std::uint32_t const local_offset = threadIdxInWarp;

        std::uint32_t const warp_offsets = alpaka::detail::warp_exclusive_sum(acc, local_offset, threadIdxInWarp);

        std::uint32_t const sum
            = threadIdxInWarp == 0 ? (warpExtent - 1) * warpExtent / 2 : (threadIdxInWarp - 1) * threadIdxInWarp / 2;

        ALPAKA_CHECK(*success, warp_offsets == sum);

        std::int32_t const parity = threadIdxInWarp & 1;

        MaskType const parity_mask = alpaka::warp::match_any(acc, mask, parity);

        std::uint32_t const warp_parity_offsets
            = alpaka::detail::warp_sparse_exclusive_sum(acc, parity_mask, local_offset, threadIdxInWarp);

        if(parity == 0)
        {
            std::uint32_t const even_sum = threadIdxInWarp < 2 ? (warpExtent / 2) * ((warpExtent / 2) - 1)
                                                               : (threadIdxInWarp / 2 - 1) * (threadIdxInWarp / 2);
            ALPAKA_CHECK(*success, warp_parity_offsets == even_sum);
        }
        else
        {
            std::uint32_t const odd_sum = threadIdxInWarp < 2 ? (warpExtent / 2) * (warpExtent / 2)
                                                              : (threadIdxInWarp / 2) * (threadIdxInWarp / 2);
            ALPAKA_CHECK(*success, warp_parity_offsets == odd_sum);
        }
    }
};

template<std::uint32_t TWarpSize, typename TAcc>
struct alpaka::trait::WarpSize<WarpAlgosTestKernel<TWarpSize>, TAcc> : std::integral_constant<std::uint32_t, TWarpSize>
{
};

TEMPLATE_LIST_TEST_CASE("warp_algos", "[warp]", alpaka::test::TestAccs)
{
    using Acc = TestType;

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(__INTEL_LLVM_COMPILER) && __INTEL_LLVM_COMPILER < 20'250'300
    if constexpr(alpaka::accMatchesTags<
                     Acc,
                     alpaka::TagCpuSycl,
                     alpaka::TagGpuSyclIntel,
                     alpaka::TagGpuSyclNvidia,
                     alpaka::TagGpuSyclAmd,
                     alpaka::TagFpgaSyclIntel,
                     alpaka::TagGenericSycl>)
    {
        WARN("Test disabled for SYCL");
        return;
    }
#else
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    auto const platform = alpaka::Platform<Acc>{};
    auto const dev = alpaka::getDevByIdx(platform, 0);
    auto const warpExtents = alpaka::getWarpSizes(dev);
    for(auto const warpExtent : warpExtents)
    {
        {
            using ExecutionFixture = alpaka::test::KernelExecutionFixture<Acc>;
            auto const gridBlockExtent = alpaka::Vec<Dim, Idx>::all(2);
            // Enforce one warp per thread block
            auto blockThreadExtent = alpaka::Vec<Dim, Idx>::ones();
            blockThreadExtent[0] = static_cast<Idx>(warpExtent);
            auto const threadElementExtent = alpaka::Vec<Dim, Idx>::ones();
            auto workDiv = typename ExecutionFixture::WorkDiv{gridBlockExtent, blockThreadExtent, threadElementExtent};
            auto fixture = ExecutionFixture{workDiv};
            if(warpExtent == 1)
            {
                REQUIRE(fixture(WarpAlgosTestKernel<1>{}));
            }
            else if(warpExtent == 4)
            {
                REQUIRE(fixture(WarpAlgosTestKernel<4>{}));
            }
            else if(warpExtent == 8)
            {
                REQUIRE(fixture(WarpAlgosTestKernel<8>{}));
            }
            else if(warpExtent == 16)
            {
                REQUIRE(fixture(WarpAlgosTestKernel<16>{}));
            }
            else if(warpExtent == 32)
            {
                REQUIRE(fixture(WarpAlgosTestKernel<32>{}));
            }
            else if(warpExtent == 64)
            {
                REQUIRE(fixture(WarpAlgosTestKernel<64>{}));
            }
        }
    }
#endif
}
