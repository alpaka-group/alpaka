/* Copyright 2022 Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/kernel/TaskKernelGenericSycl.hpp"
#include "alpaka/platform/PlatformFpgaSyclIntel.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_FPGA)

namespace alpaka
{
    template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    using TaskKernelFpgaSyclIntel
        = TaskKernelGenericSycl<detail::IntelFpgaSelector, AccFpgaSyclIntel<TDim, TIdx>, TDim, TIdx, TKernelFnObj, TArgs...>;

} // namespace alpaka

#endif
