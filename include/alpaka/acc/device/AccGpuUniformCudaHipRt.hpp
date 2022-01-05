/* Copyright 2022 Andrea Bocci
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

// Host API: base classes, specialized traits, implementation details.
#    include <alpaka/acc/AccGpuUniformCudaHipRt.hpp>

// Device code.
#    include <alpaka/atomic/device/AtomicUniformCudaHipBuiltIn.hpp>
#    include <alpaka/block/shared/dyn/device/BlockSharedMemDynUniformCudaHipBuiltIn.hpp>
#    include <alpaka/block/shared/st/device/BlockSharedMemStUniformCudaHipBuiltIn.hpp>
#    include <alpaka/block/sync/device/BlockSyncUniformCudaHipBuiltIn.hpp>
#    include <alpaka/core/DeviceOnly.hpp>
#    include <alpaka/idx/bt/device/IdxBtUniformCudaHipBuiltIn.hpp>
#    include <alpaka/idx/gb/device/IdxGbUniformCudaHipBuiltIn.hpp>
#    include <alpaka/intrinsic/device/IntrinsicUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/device/MathUniformCudaHipBuiltIn.hpp>
#    include <alpaka/mem/fence/device/MemFenceUniformCudaHipBuiltIn.hpp>
#    include <alpaka/rand/device/RandUniformCudaHipRand.hpp>
#    include <alpaka/time/device/TimeUniformCudaHipBuiltIn.hpp>
#    include <alpaka/warp/device/WarpUniformCudaHipBuiltIn.hpp>
#    include <alpaka/workdiv/device/WorkDivUniformCudaHipBuiltIn.hpp>

#endif
