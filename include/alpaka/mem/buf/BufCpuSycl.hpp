/* Copyright 2024 Jan Stephan, Luca Ferragina, Andrea Bocci, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/mem/buf/BufGenericSycl.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_CPU)

namespace alpaka
{
    template<typename TElem, typename TDim, typename TIdx>
    using BufCpuSycl = BufGenericSycl<TElem, TDim, TIdx, TagCpuSycl>;
} // namespace alpaka

#endif
