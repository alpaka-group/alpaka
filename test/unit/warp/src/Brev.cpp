/* SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>
#include <alpaka/warp/Traits.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>

struct BrevSingleThreadWarpTestKernel
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        if constexpr(alpaka::Dim<TAcc>::value > 0)
            ALPAKA_CHECK(*success, alpaka::warp::getSize(acc) == 1);
        std::int32_t const match_val = 0;
        ALPAKA_CHECK(*success, alpaka::warp::brev(acc, 1) == 1);
    }
};

template<std::uint32_t TWarpSize>
struct BrevMultipleThreadWarpTestKernel
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
        auto const threadIdxInWarp = std::int32_t(alpaka::mapIdx<1u>(localThreadIdx, blockExtent)[0]);

        auto const mask = alpaka::warp::activemask(acc);
        using MaskType = decltype(mask);

        ALPAKA_CHECK(*success, warpExtent > 1);

        ALPAKA_CHECK(*success, alpaka::warp::brev(acc, mask) == mask);

        std::int32_t const ballot_val = threadIdxInWarp & 1;

        MaskType const odd_mask = alpaka::warp::ballot(acc, mask, ballot_val == 1);

        MaskType const brev_mask = alpaka::warp::brev(acc, odd_mask);

        if(threadIdxInWarp % 2 == 0)
        {
            MaskType const even_mask = alpaka::warp::activemask(acc);
            ALPAKA_CHECK(*success, brev_mask == even_mask);
        }
    }
};

template<std::uint32_t TWarpSize, typename TAcc>
struct alpaka::trait::WarpSize<BrevMultipleThreadWarpTestKernel<TWarpSize>, TAcc>
    : std::integral_constant<std::uint32_t, TWarpSize>
{
};

TEMPLATE_LIST_TEST_CASE("brev", "[warp]", alpaka::test::TestAccs)
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
        auto const scalar = Dim::value == 0 || warpExtent == 1;
        if(scalar)
        {
            alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::all(4));
            REQUIRE(fixture(BrevSingleThreadWarpTestKernel{}));
        }
        else
        {
            using ExecutionFixture = alpaka::test::KernelExecutionFixture<Acc>;
            auto const gridBlockExtent = alpaka::Vec<Dim, Idx>::all(2);
            // Enforce one warp per thread block
            auto blockThreadExtent = alpaka::Vec<Dim, Idx>::ones();
            blockThreadExtent[0] = static_cast<Idx>(warpExtent);
            auto const threadElementExtent = alpaka::Vec<Dim, Idx>::ones();
            auto workDiv = typename ExecutionFixture::WorkDiv{gridBlockExtent, blockThreadExtent, threadElementExtent};
            auto fixture = ExecutionFixture{workDiv};
            if(warpExtent == 4)
            {
                REQUIRE(fixture(BrevMultipleThreadWarpTestKernel<4>{}));
            }
            else if(warpExtent == 8)
            {
                REQUIRE(fixture(BrevMultipleThreadWarpTestKernel<8>{}));
            }
            else if(warpExtent == 16)
            {
                REQUIRE(fixture(BrevMultipleThreadWarpTestKernel<16>{}));
            }
            else if(warpExtent == 32)
            {
                REQUIRE(fixture(BrevMultipleThreadWarpTestKernel<32>{}));
            }
            else if(warpExtent == 64)
            {
                REQUIRE(fixture(BrevMultipleThreadWarpTestKernel<64>{}));
            }
        }
    }
#endif
}
