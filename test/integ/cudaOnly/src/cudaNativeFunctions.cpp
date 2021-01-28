/* Copyright 2019-2021 Axel Huebl, Benjamin Worpitz, René Widera, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>

#include <catch2/catch.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ONLY_MODE) && defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA

//! Native CUDA function.
#    if BOOST_COMP_CLANG
#        pragma clang diagnostic push
#        pragma clang diagnostic ignored "-Wmissing-prototypes"
#    endif
__device__ auto userDefinedThreadFence() -> void
{
    __threadfence();
}
#    if BOOST_COMP_CLANG
#        pragma clang diagnostic pop
#    endif

class CudaOnlyTestKernel
{
public:
    template<typename TAcc, typename TMemoryHandle>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        alpaka::experimental::
            Accessor<TMemoryHandle, bool, alpaka::Idx<TAcc>, 1, alpaka::experimental::WriteAccess> const success) const
        -> void
    {
        alpaka::ignore_unused(acc);

        // We should be able to call some native CUDA functions when ALPAKA_ACC_GPU_CUDA_ONLY_MODE is enabled.
        __threadfence_block();
        userDefinedThreadFence();
        __threadfence_system();

        success[0] = true;
    }
};


TEST_CASE("cudaOnlyModeWorking", "[cudaOnly]")
{
    using TAcc = alpaka::AccGpuCudaRt<alpaka::DimInt<1u>, std::uint32_t>;
    using Dim = alpaka::Dim<TAcc>;
    using Idx = alpaka::Idx<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(alpaka::Vec<Dim, Idx>::ones());

    CudaOnlyTestKernel kernel;

    REQUIRE(fixture(kernel));
}

#endif
