/* Copyright 2023 Jan Stephan, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/DevGenericSycl.hpp"
#include "alpaka/platform/PlatformCpuSycl.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_CPU)

namespace alpaka
{
    using DevCpuSycl = DevGenericSycl<PlatformCpuSycl>;

    namespace trait
    {
        template<>
        struct MemVisibility<DevCpuSycl>
        {
            using type = alpaka::MemVisibleCPU;
        };
    } // namespace trait
} // namespace alpaka

#endif
