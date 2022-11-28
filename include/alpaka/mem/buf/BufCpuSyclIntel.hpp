/* Copyright 2022 Jan Stephan, Luca Ferragina
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/DevCpuSyclIntel.hpp"
#include "alpaka/mem/buf/BufGenericSycl.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI) && defined(ALPAKA_SYCL_ONEAPI_CPU)

namespace alpaka
{
    template<typename TElem, typename TDim, typename TIdx>
    using BufCpuSyclIntel = BufGenericSycl<TElem, TDim, TIdx, PltfCpuSyclIntel>;
}

#endif
