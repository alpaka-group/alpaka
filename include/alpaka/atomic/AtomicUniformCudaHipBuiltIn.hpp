/* Copyright 2022 Benjamin Worpitz, René Widera, Andrea Bocci
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

namespace alpaka
{
    //! The GPU CUDA/HIP accelerator atomic ops.
    //
    //  Atomics can be used in the hierarchy level grids, blocks and threads.
    //  Atomics are not guaranteed to be save between devices.
    class AtomicUniformCudaHipBuiltIn
    {
    };
} // namespace alpaka

#endif
