/* Copyright 2022 Andrea Bocci
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

// UniformCudaHip implementation
#    define ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE cuda
#    include <alpaka/event/EventUniformCudaHipRt.hpp>
#    undef ALPAKA_UNIFORM_CUDA_HIP_RT_NAMESPACE

namespace alpaka
{
    using EventCudaRt = cuda::EventUniformCudaHipRt;
}

#endif // ALPAKA_ACC_GPU_CUDA_ENABLED
