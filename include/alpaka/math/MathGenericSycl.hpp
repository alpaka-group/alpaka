/* Copyright 2021 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#include <alpaka/core/Common.hpp>
#include <alpaka/math/abs/AbsGenericSycl.hpp>
#include <alpaka/math/acos/AcosGenericSycl.hpp>
#include <alpaka/math/asin/AsinGenericSycl.hpp>
#include <alpaka/math/atan/AtanGenericSycl.hpp>
#include <alpaka/math/atan2/Atan2GenericSycl.hpp>
#include <alpaka/math/cbrt/CbrtGenericSycl.hpp>
#include <alpaka/math/ceil/CeilGenericSycl.hpp>
#include <alpaka/math/cos/CosGenericSycl.hpp>
#include <alpaka/math/erf/ErfGenericSycl.hpp>
#include <alpaka/math/exp/ExpGenericSycl.hpp>
#include <alpaka/math/floor/FloorGenericSycl.hpp>
#include <alpaka/math/fmod/FmodGenericSycl.hpp>
#include <alpaka/math/log/LogGenericSycl.hpp>
#include <alpaka/math/max/MaxGenericSycl.hpp>
#include <alpaka/math/min/MinGenericSycl.hpp>
#include <alpaka/math/pow/PowGenericSycl.hpp>
#include <alpaka/math/remainder/RemainderGenericSycl.hpp>
#include <alpaka/math/round/RoundGenericSycl.hpp>
#include <alpaka/math/rsqrt/RsqrtGenericSycl.hpp>
#include <alpaka/math/sin/SinGenericSycl.hpp>
#include <alpaka/math/sincos/SinCosGenericSycl.hpp>
#include <alpaka/math/sqrt/SqrtGenericSycl.hpp>
#include <alpaka/math/tan/TanGenericSycl.hpp>
#include <alpaka/math/trunc/TruncGenericSycl.hpp>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The mathematical operation specifics.
    namespace math
    {
        //#############################################################################
        //! The standard library math trait specializations.
        class MathGenericSycl :
            public AbsGenericSycl,
            public AcosGenericSycl,
            public AsinGenericSycl,
            public AtanGenericSycl,
            public Atan2GenericSycl,
            public CbrtGenericSycl,
            public CeilGenericSycl,
            public CosGenericSycl,
            public ErfGenericSycl,
            public ExpGenericSycl,
            public FloorGenericSycl,
            public FmodGenericSycl,
            public LogGenericSycl,
            public MaxGenericSycl,
            public MinGenericSycl,
            public PowGenericSycl,
            public RemainderGenericSycl,
            public RoundGenericSycl,
            public RsqrtGenericSycl,
            public SinGenericSycl,
            public SinCosGenericSycl,
            public SqrtGenericSycl,
            public TanGenericSycl,
            public TruncGenericSycl
        {};
    }
}

#endif
