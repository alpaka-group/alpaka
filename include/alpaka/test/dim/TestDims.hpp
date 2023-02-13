/* Copyright 2022 Benjamin Worpitz, Andrea Bocci, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>

#include <tuple>

namespace alpaka::test
{
    //! A std::tuple holding dimensions.
    using TestDims = std::tuple<
        DimInt<0u>,
        DimInt<1u>,
        DimInt<2u>,
        DimInt<3u>
    // The CUDA & HIP accelerators do not currently support 4D buffers and 4D acceleration.
#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !defined(ALPAKA_ACC_SYCL_ENABLED)
        ,
        DimInt<4u>
#endif
        >;
} // namespace alpaka::test
