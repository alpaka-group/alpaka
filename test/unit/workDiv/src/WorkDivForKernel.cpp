/* Copyright 2022 Sergei Bastrakov, Jan Stephan, Bernhard Manfred Gruber, Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/acc/AccDevProps.hpp>
#include <alpaka/acc/AccGpuHipRt.hpp>
#include <alpaka/acc/AccGpuUniformCudaHipRt.hpp>
#include <alpaka/core/ApiHipRt.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/workdiv/WorkDivHelpers.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

namespace
{
    template<typename TAcc>
    auto getWorkDivKernel()
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

    struct HelloWorldKernel
    {
        template<typename TAcc>
        ALPAKA_FN_ACC auto operator()(TAcc const& acc) const -> void
        {
            using Dim = alpaka::Dim<TAcc>;
            using Idx = alpaka::Idx<TAcc>;
            using Vec = alpaka::Vec<Dim, Idx>;
            using Vec1 = alpaka::Vec<alpaka::DimInt<1u>, Idx>;

            // In the most cases the parallel work distibution depends
            // on the current index of a thread and how many threads
            // exist overall. These information can be obtained by
            // getIdx() and getWorkDiv(). In this example these
            // values are obtained for a global scope.
            Vec const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            Vec const globalThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

            // Map the three dimensional thread index into a
            // one dimensional thread index space. We call it
            // linearize the thread index.
            Vec1 const linearizedGlobalThreadIdx = alpaka::mapIdx<1u>(globalThreadIdx, globalThreadExtent);

            // Each thread prints a hello world to the terminal
            // together with the global index of the thread in
            // each dimension and the linearized global index.
            // Mind, that alpaka uses the mathematical index
            // order [z][y][x] where the last index is the fast one.
            printf(
                "[z:%u, y:%u, x:%u][linear:%u] Hello World\n",
                static_cast<unsigned>(globalThreadIdx[0u]),
                static_cast<unsigned>(globalThreadIdx[1u]),
                static_cast<unsigned>(globalThreadIdx[2u]),
                static_cast<unsigned>(linearizedGlobalThreadIdx[0u]));
        }
    };
} // namespace

TEMPLATE_LIST_TEST_CASE("getValidWorkDivKernel", "[workDivKernel]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    // Note: getValidWorkDiv() is called inside getWorkDiv
    std::ignore = getWorkDivKernel<Acc>();
}

TEMPLATE_LIST_TEST_CASE("enqueueWithValidWorkDiv.1D.withIdx", "[workDivKernel]", alpaka::test::TestAccs)
{
#if defined(ALPAKA_ACC_GPU_HIP_ENABLED) || defined(ALPAKA_ACC_GPU_CUDA_ENABLED)

#    if defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    using TApi = alpaka::ApiHipRt<alpaka::Dim<Acc>, alpaka::Idx<Acc>>;
#    elif defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    using TApi = alpaka::ApiCudaRt<alpaka::Dim<Acc>, alpaka::Idx<Acc>>;
#    endif
    using Acc = TestType;
    using Idx = alpaka::Idx<Acc>;
    using Dim = alpaka::Dim<Acc>;
    using Vec = alpaka::Vec<Dim, Idx>;
    if constexpr(Dim::value == 1)
    {
        auto const platform = alpaka::Platform<Acc>{};
        auto const dev = alpaka::getDevByIdx(platform, 0);
        // test that we can call getValidWorkDiv with the Idx type directly instead of a Vec
        alpaka::enqueueWithValidWorkDiv<TApi, Acc>(dev, HelloWorldKernel{}, Vec{256}, Vec{13});
        // CHECK(alpaka::enqueueWithValidWorkDiv<Acc>(dev, Idx{256}, Idx{13}));
    }
#endif
}
