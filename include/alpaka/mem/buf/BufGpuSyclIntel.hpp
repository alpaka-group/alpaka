/* Copyright 2023 Jan Stephan, Luca Ferragina
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/DevGpuSyclIntel.hpp"
#include "alpaka/mem/buf/BufGenericSycl.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_GPU)

namespace alpaka
{
    template<typename TElem, typename TDim, typename TIdx>
    using BufGpuSyclIntel = BufGenericSycl<TElem, TDim, TIdx, PlatformGpuSyclIntel>;

    namespace trait
    {
        template<typename TElem, typename TDim, typename TIdx>
        struct MemVisibility<BufGpuSyclIntel<TElem, TDim, TIdx>>
        {
            using type = std::tuple<alpaka::MemVisibleGpuSyclIntel>;
        };
    } // namespace trait
} // namespace alpaka

#endif
