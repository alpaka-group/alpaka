/* Copyright 2022 Benjamin Worpitz, Andrea Bocci
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/core/DeviceOnly.hpp>
#    include <alpaka/time/TimeUniformCudaHipBuiltIn.hpp>

namespace alpaka
{
    namespace traits
    {
        //! The CUDA built-in clock operation.
        template<>
        struct Clock<TimeUniformCudaHipBuiltIn>
        {
            __device__ static auto clock(TimeUniformCudaHipBuiltIn const&) -> std::uint64_t
            {
                // This can be converted to a wall-clock time in seconds by dividing through the shader clock rate
                // given by uniformCudaHipDeviceProp::clockRate. This clock rate is double the main clock rate on Fermi
                // and older cards.
                return static_cast<std::uint64_t>(clock64());
            }
        };
    } // namespace traits
} // namespace alpaka

#endif
