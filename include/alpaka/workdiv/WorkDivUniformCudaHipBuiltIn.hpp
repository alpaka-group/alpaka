/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Andrea Bocci
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/vec/Vec.hpp>
#    include <alpaka/workdiv/Traits.hpp>

namespace alpaka
{
    //! The GPU CUDA/HIP accelerator work division.
    template<typename TDim, typename TIdx>
    class WorkDivUniformCudaHipBuiltIn
        : public concepts::Implements<ConceptWorkDiv, WorkDivUniformCudaHipBuiltIn<TDim, TIdx>>
    {
    public:
        ALPAKA_FN_HOST_ACC WorkDivUniformCudaHipBuiltIn(Vec<TDim, TIdx> const& threadElemExtent)
            : m_threadElemExtent(threadElemExtent)
        {
        }

        // \TODO: Optimize! Add WorkDivUniformCudaHipBuiltInNoElems that has no member m_threadElemExtent as well as
        // AccGpuUniformCudaHipRtNoElems. Use it instead of AccGpuUniformCudaHipRt if the thread element extent is one
        // to reduce the register usage.
        Vec<TDim, TIdx> const& m_threadElemExtent;
    };
} // namespace alpaka

#endif
