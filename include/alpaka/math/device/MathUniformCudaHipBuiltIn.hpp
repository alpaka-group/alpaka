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

#    include <alpaka/math/abs/device/AbsUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/acos/device/AcosUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/asin/device/AsinUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/atan/device/AtanUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/atan2/device/Atan2UniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/cbrt/device/CbrtUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/ceil/device/CeilUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/cos/device/CosUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/erf/device/ErfUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/exp/device/ExpUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/floor/device/FloorUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/fmod/device/FmodUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/isfinite/device/IsfiniteUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/isinf/device/IsinfUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/isnan/device/IsnanUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/log/device/LogUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/max/device/MaxUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/min/device/MinUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/pow/device/PowUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/remainder/device/RemainderUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/round/device/RoundUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/rsqrt/device/RsqrtUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/sin/device/SinUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/sincos/device/SinCosUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/sqrt/device/SqrtUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/tan/device/TanUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/trunc/device/TruncUniformCudaHipBuiltIn.hpp>

#endif
