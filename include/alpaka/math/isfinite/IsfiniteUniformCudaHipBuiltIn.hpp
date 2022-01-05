/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Bert Wesarg, Jeffrey Kelling, Andrea Bocci
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/core/CudaHipMath.hpp>
#    include <alpaka/math/isfinite/Traits.hpp>

namespace alpaka
{
    namespace math
    {
        //! The CUDA built in isfinite.
        class IsfiniteUniformCudaHipBuiltIn
            : public concepts::Implements<ConceptMathIsfinite, IsfiniteUniformCudaHipBuiltIn>
        {
        };
    } // namespace math
} // namespace alpaka

#endif
