/* Copyright 2022 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/ApiHipRt.hpp"
#include "alpaka/mem/Visibility.hpp"
#include "alpaka/platform/PlatformUniformCudaHipRt.hpp"

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

namespace alpaka
{
    //! The HIP RT platform.
    using PlatformHipRt = PlatformUniformCudaHipRt<ApiHipRt>;

    namespace trait
    {
        template<>
        struct MemVisibility<PlatformHipRt>
        {
            using type = alpaka::MemVisibleGpuHipRt;
        };
    } // namespace trait
} // namespace alpaka

#endif // ALPAKA_ACC_GPU_HIP_ENABLED
