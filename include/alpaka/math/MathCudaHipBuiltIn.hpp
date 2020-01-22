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
        #if BOOST_COMP_HCC
            #include <math_functions.h>
        #else
            #include <math_functions.hpp>
        #endif
    #endif
    #if !BOOST_LANG_HIP
        #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
    #endif
#endif

#include <alpaka/math/abs/AbsCudaHipBuiltIn.hpp>
#include <alpaka/math/acos/AcosCudaHipBuiltIn.hpp>
#include <alpaka/math/asin/AsinCudaHipBuiltIn.hpp>
#include <alpaka/math/atan/AtanCudaHipBuiltIn.hpp>
#include <alpaka/math/atan2/Atan2CudaHipBuiltIn.hpp>
#include <alpaka/math/cbrt/CbrtCudaHipBuiltIn.hpp>
#include <alpaka/math/ceil/CeilCudaHipBuiltIn.hpp>
#include <alpaka/math/cos/CosCudaHipBuiltIn.hpp>
#include <alpaka/math/erf/ErfCudaHipBuiltIn.hpp>
#include <alpaka/math/exp/ExpCudaHipBuiltIn.hpp>
#include <alpaka/math/floor/FloorCudaHipBuiltIn.hpp>
#include <alpaka/math/fmod/FmodCudaHipBuiltIn.hpp>
#include <alpaka/math/log/LogCudaHipBuiltIn.hpp>
#include <alpaka/math/max/MaxCudaHipBuiltIn.hpp>
#include <alpaka/math/min/MinCudaHipBuiltIn.hpp>
#include <alpaka/math/pow/PowCudaHipBuiltIn.hpp>
#include <alpaka/math/remainder/RemainderCudaHipBuiltIn.hpp>
#include <alpaka/math/round/RoundCudaHipBuiltIn.hpp>
#include <alpaka/math/rsqrt/RsqrtCudaHipBuiltIn.hpp>
#include <alpaka/math/sin/SinCudaHipBuiltIn.hpp>
#include <alpaka/math/sincos/SinCosCudaHipBuiltIn.hpp>
#include <alpaka/math/sqrt/SqrtCudaHipBuiltIn.hpp>
#include <alpaka/math/tan/TanCudaHipBuiltIn.hpp>
#include <alpaka/math/trunc/TruncCudaHipBuiltIn.hpp>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The mathematical operation specifics.
    namespace math
    {
        //#############################################################################
        //! The standard library math trait specializations.
        class MathCudaHipBuiltIn :
            public AbsCudaHipBuiltIn,
            public AcosCudaHipBuiltIn,
            public AsinCudaHipBuiltIn,
            public AtanCudaHipBuiltIn,
            public Atan2CudaHipBuiltIn,
            public CbrtCudaHipBuiltIn,
            public CeilCudaHipBuiltIn,
            public CosCudaHipBuiltIn,
            public ErfCudaHipBuiltIn,
            public ExpCudaHipBuiltIn,
            public FloorCudaHipBuiltIn,
            public FmodCudaHipBuiltIn,
            public LogCudaHipBuiltIn,
            public MaxCudaHipBuiltIn,
            public MinCudaHipBuiltIn,
            public PowCudaHipBuiltIn,
            public RemainderCudaHipBuiltIn,
            public RoundCudaHipBuiltIn,
            public RsqrtCudaHipBuiltIn,
            public SinCudaHipBuiltIn,
            public SinCosCudaHipBuiltIn,
            public SqrtCudaHipBuiltIn,
            public TanCudaHipBuiltIn,
            public TruncCudaHipBuiltIn
        {};
    }
}

#endif
