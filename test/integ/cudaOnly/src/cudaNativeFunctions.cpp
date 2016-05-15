/**
 * \file
 * Copyright 2016 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * alpaka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * alpaka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with alpaka.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>   // alpaka::test::KernelExecutionFixture

#include <boost/test/unit_test.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ONLY_MODE) && defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA

//-----------------------------------------------------------------------------
//! Native CUDA function.
//-----------------------------------------------------------------------------
__device__ auto userDefinedThreadFence()
-> void
{
    __threadfence();
}

//#############################################################################
//!
//#############################################################################
class CudaOnlyTestKernel
{
public:
    //-----------------------------------------------------------------------------
    //!
    //-----------------------------------------------------------------------------
    template<
        typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc) const
    -> void
    {
        // We should be able to call some native CUDA functions when ALPAKA_ACC_GPU_CUDA_ONLY_MODE is enabled.
        __threadfence_block();
        userDefinedThreadFence();
        __threadfence_system();
    }
};

BOOST_AUTO_TEST_SUITE(cudaOnly)

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(cudaOnlyModeWorking)
{
    using TAcc = alpaka::acc::AccGpuCudaRt<alpaka::dim::DimInt<1u>, std::uint32_t>;
    using Dim = alpaka::dim::Dim<TAcc>;
    using Size = alpaka::size::Size<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::Vec<Dim, Size>::ones());

    CudaOnlyTestKernel kernel;

    BOOST_REQUIRE_EQUAL(
        true,
        fixture(
            kernel));
}

BOOST_AUTO_TEST_SUITE_END()

#endif
