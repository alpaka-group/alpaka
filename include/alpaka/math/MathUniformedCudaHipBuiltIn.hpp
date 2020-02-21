/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#include <alpaka/core/BoostPredef.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    #include <cuda_runtime.h>
    #if !BOOST_LANG_CUDA
        #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
    #endif
#endif

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED)

    #if BOOST_COMP_NVCC >= BOOST_VERSION_NUMBER(9, 0, 0)
        #include <cuda_runtime_api.h>
    #else
        #if BOOST_COMP_HCC || BOOST_COMP_HIP
            #include <math_functions.h>
        #else
            #include <math_functions.hpp>
        #endif
    #endif
    #if !BOOST_LANG_HIP
        #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
    #endif
#endif

#include <alpaka/math/abs/AbsUniformedCudaHipBuiltIn.hpp>
#include <alpaka/math/acos/AcosUniformedCudaHipBuiltIn.hpp>
#include <alpaka/math/asin/AsinUniformedCudaHipBuiltIn.hpp>
#include <alpaka/math/atan/AtanUniformedCudaHipBuiltIn.hpp>
#include <alpaka/math/atan2/Atan2UniformedCudaHipBuiltIn.hpp>
#include <alpaka/math/cbrt/CbrtUniformedCudaHipBuiltIn.hpp>
#include <alpaka/math/ceil/CeilUniformedCudaHipBuiltIn.hpp>
#include <alpaka/math/cos/CosUniformedCudaHipBuiltIn.hpp>
#include <alpaka/math/erf/ErfUniformedCudaHipBuiltIn.hpp>
#include <alpaka/math/exp/ExpUniformedCudaHipBuiltIn.hpp>
#include <alpaka/math/floor/FloorUniformedCudaHipBuiltIn.hpp>
#include <alpaka/math/fmod/FmodUniformedCudaHipBuiltIn.hpp>
#include <alpaka/math/log/LogUniformedCudaHipBuiltIn.hpp>
#include <alpaka/math/max/MaxUniformedCudaHipBuiltIn.hpp>
#include <alpaka/math/min/MinUniformedCudaHipBuiltIn.hpp>
#include <alpaka/math/pow/PowUniformedCudaHipBuiltIn.hpp>
#include <alpaka/math/remainder/RemainderUniformedCudaHipBuiltIn.hpp>
#include <alpaka/math/round/RoundUniformedCudaHipBuiltIn.hpp>
#include <alpaka/math/rsqrt/RsqrtUniformedCudaHipBuiltIn.hpp>
#include <alpaka/math/sin/SinUniformedCudaHipBuiltIn.hpp>
#include <alpaka/math/sincos/SinCosUniformedCudaHipBuiltIn.hpp>
#include <alpaka/math/sqrt/SqrtUniformedCudaHipBuiltIn.hpp>
#include <alpaka/math/tan/TanUniformedCudaHipBuiltIn.hpp>
#include <alpaka/math/trunc/TruncUniformedCudaHipBuiltIn.hpp>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The mathematical operation specifics.
    namespace math
    {
        //#############################################################################
        //! The standard library math trait specializations.
        class MathUniformedCudaHipBuiltIn :
            public AbsUniformedCudaHipBuiltIn,
            public AcosUniformedCudaHipBuiltIn,
            public AsinUniformedCudaHipBuiltIn,
            public AtanUniformedCudaHipBuiltIn,
            public Atan2UniformedCudaHipBuiltIn,
            public CbrtUniformedCudaHipBuiltIn,
            public CeilUniformedCudaHipBuiltIn,
            public CosUniformedCudaHipBuiltIn,
            public ErfUniformedCudaHipBuiltIn,
            public ExpUniformedCudaHipBuiltIn,
            public FloorUniformedCudaHipBuiltIn,
            public FmodUniformedCudaHipBuiltIn,
            public LogUniformedCudaHipBuiltIn,
            public MaxUniformedCudaHipBuiltIn,
            public MinUniformedCudaHipBuiltIn,
            public PowUniformedCudaHipBuiltIn,
            public RemainderUniformedCudaHipBuiltIn,
            public RoundUniformedCudaHipBuiltIn,
            public RsqrtUniformedCudaHipBuiltIn,
            public SinUniformedCudaHipBuiltIn,
            public SinCosUniformedCudaHipBuiltIn,
            public SqrtUniformedCudaHipBuiltIn,
            public TanUniformedCudaHipBuiltIn,
            public TruncUniformedCudaHipBuiltIn
        {};
    }
}

#endif
