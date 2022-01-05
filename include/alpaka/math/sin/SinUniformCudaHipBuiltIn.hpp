/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Bert Wesarg, Valentin Gehrke, Jan Stephan, Andrea Bocci
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/math/sin/Traits.hpp>

namespace alpaka
{
    namespace math
    {
        //! The CUDA sin.
        class SinUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathSin, SinUniformCudaHipBuiltIn>
        {
        };
    } // namespace math
} // namespace alpaka

#endif
