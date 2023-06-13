/* Copyright 2022 Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI) && defined(ALPAKA_SYCL_ONEAPI_GPU)

#    include "alpaka/dev/DevGpuSyclIntel.hpp"
#    include "alpaka/mem/buf/BufGenericSycl.hpp"

namespace alpaka
{
    template<typename TElem, typename TDim, typename TIdx>
    using BufGpuSyclIntel = BufGenericSycl<TElem, TDim, TIdx, DevGpuSyclIntel>;
}

#endif
