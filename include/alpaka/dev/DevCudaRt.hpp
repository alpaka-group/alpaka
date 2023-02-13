/* Copyright 2022 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#    include <alpaka/core/ApiCudaRt.hpp>
#    include <alpaka/dev/DevUniformCudaHipRt.hpp>

namespace alpaka
{
    //! The CUDA RT device handle.
    using DevCudaRt = DevUniformCudaHipRt<ApiCudaRt>;
} // namespace alpaka

#endif // ALPAKA_ACC_GPU_CUDA_ENABLED
