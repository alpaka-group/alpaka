/* Copyright 2022 Benjamin Worpitz, Matthias Werner, Jan Stephan, Andrea Bocci
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
#    include <alpaka/math/sincos/Traits.hpp>

namespace alpaka
{
    namespace math
    {
        //! The CUDA sincos.
        class SinCosUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathSinCos, SinCosUniformCudaHipBuiltIn>
        {
        };
    } // namespace math
} // namespace alpaka

#endif
