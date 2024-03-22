/* Copyright 2022 Sergei Bastrakov, Jan Stephan, Bernhard Manfred Gruber, Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/acc/AccDevProps.hpp>
#include <alpaka/acc/AccGpuHipRt.hpp>
#include <alpaka/acc/AccGpuUniformCudaHipRt.hpp>
#include <alpaka/core/ApiHipRt.hpp>
// #include <alpaka/kernel/TaskKernelGpuUniformCudaHipRt.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/workdiv/WorkDivHelpers.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
namespace alpaka::detail
{
#    if BOOST_COMP_CLANG
#        pragma clang diagnostic push
#        pragma clang diagnostic ignored "-Wunused-template"
#    endif
    //! The GPU CUDA/HIP kernel entry point.
    // \NOTE: 'A __global__ function or function template cannot have a trailing return type.'
    // We have put the function into a shallow namespace and gave it a short name, so the mangled name in the
    // profiler (e.g. ncu) is as shorter as possible.
    template<typename TKernelFnObj, typename TApi, typename TAcc, typename TDim, typename TIdx, typename... TArgs>
    __global__ void gpuKernel(Vec<TDim, TIdx> const threadElemExtent, TKernelFnObj const kernelFnObj, TArgs... args)
    {
#    if BOOST_ARCH_PTX && (BOOST_ARCH_PTX < BOOST_VERSION_NUMBER(2, 0, 0))
#        error "Device capability >= 2.0 is required!"
#    endif

        TAcc const acc(threadElemExtent);

// with clang it is not possible to query std::result_of for a pure device lambda created on the host side
#    if !(BOOST_COMP_CLANG_CUDA && BOOST_COMP_CLANG)
        static_assert(
            std::is_same_v<decltype(kernelFnObj(const_cast<TAcc const&>(acc), args...)), void>,
            "The TKernelFnObj is required to return void!");
#    endif
        kernelFnObj(const_cast<TAcc const&>(acc), args...);
    }
#    if BOOST_COMP_CLANG
#        pragma clang diagnostic pop
#    endif
} // namespace alpaka::detail
#endif

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
    [[maybe_unused]] HelloWorldKernel kernel;
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
        alpaka::enqueueWithValidWorkDiv<TApi, Acc, decltype(dev), HelloWorldKernel>(dev, kernel, Vec{256}, Vec{13});
        // CHECK(alpaka::enqueueWithValidWorkDiv<Acc>(dev, Idx{256}, Idx{13}));
    }
#endif
}
