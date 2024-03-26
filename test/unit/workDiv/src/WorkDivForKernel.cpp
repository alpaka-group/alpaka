/* Copyright 2022 Sergei Bastrakov, Jan Stephan, Bernhard Manfred Gruber, Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/acc/AccDevProps.hpp>
#include <alpaka/acc/AccGpuHipRt.hpp>
#include <alpaka/acc/AccGpuUniformCudaHipRt.hpp>
#include <alpaka/core/ApiCudaRt.hpp>
#include <alpaka/core/ApiHipRt.hpp>
#include <alpaka/kernel/TaskKernelGpuUniformCudaHipRt.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/workdiv/WorkDivHelpers.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

namespace
{
    struct TestKernel
    {
        template<typename TAcc>
        [[maybe_unused]] ALPAKA_FN_ACC auto operator()(TAcc const& acc) const -> void
        {
            using Dim = alpaka::Dim<TAcc>;
            using Idx = alpaka::Idx<TAcc>;
            using Vec = alpaka::Vec<Dim, Idx>;
            using Vec1 = alpaka::Vec<alpaka::DimInt<1u>, Idx>;

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
                "[z:%u, y:%u, x:%u][linear:%u] Test Kernel\n",
                static_cast<unsigned>(globalThreadIdx[0u]),
                static_cast<unsigned>(globalThreadIdx[1u]),
                static_cast<unsigned>(globalThreadIdx[2u]),
                static_cast<unsigned>(linearizedGlobalThreadIdx[0u]));
        }
    };
} // namespace

TEMPLATE_LIST_TEST_CASE("getValidWorkDivForKernel.1D.withIdx", "[workDivKernel]", alpaka::test::TestAccs)
{
#    if defined(ALPAKA_ACC_GPU_HIP_ENABLED) || defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    using Acc = TestType;
    using Idx = alpaka::Idx<Acc>;
    using Dim = alpaka::Dim<Acc>;
    [[maybe_unused]] TestKernel kernel;


#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    using TApi = alpaka::ApiCudaRt<Dim, Idx>;
#        endif

#        if defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    using TApi = alpaka::ApiHipRt<Dim, Idx>;
#        endif

    using Vec = alpaka::Vec<Dim, Idx>;
    if constexpr(Dim::value == 1)
    {
        auto const platform = alpaka::Platform<Acc>{};
        auto const dev = alpaka::getDevByIdx(platform, 0);

        alpaka::getWorkDivForKernel<TApi, Acc, decltype(dev), TestKernel>(dev, kernel, Vec{256}, Vec{13});
        // CHECK(alpaka::getValidWorkDivForKernel<Acc>(dev, Idx{256}, Idx{13}));
    }
#    endif
}
#endif
