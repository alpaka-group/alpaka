/* Copyright 2022-2021 Sergei Bastrakov, David M. Rogers, Andrea Bocci
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/warp/Traits.hpp>

namespace alpaka
{
    namespace warp
    {
        //! The GPU CUDA/HIP warp.
        class WarpUniformCudaHipBuiltIn : public concepts::Implements<ConceptWarp, WarpUniformCudaHipBuiltIn>
        {
        };
    } // namespace warp
} // namespace alpaka

#endif
