/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera, Andrea Bocci
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
#    include <alpaka/idx/Traits.hpp>

namespace alpaka
{
    namespace bt
    {
        //! The CUDA/HIP accelerator ND index provider.
        template<typename TDim, typename TIdx>
        class IdxBtUniformCudaHipBuiltIn
            : public concepts::Implements<ConceptIdxBt, IdxBtUniformCudaHipBuiltIn<TDim, TIdx>>
        {
        };
    } // namespace bt
} // namespace alpaka

#endif
