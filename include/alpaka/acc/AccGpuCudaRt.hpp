/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 *
 * SPDX-FileContributor: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-FileContributor: Simeon Ehrig <s.ehrig@hzdr.de>
 * SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/AccGpuUniformCudaHipRt.hpp"
#include "alpaka/acc/Tag.hpp"
#include "alpaka/core/ApiCudaRt.hpp"

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

namespace alpaka
{
    template<typename TDim, typename TIdx>
    using AccGpuCudaRt = AccGpuUniformCudaHipRt<ApiCudaRt, TDim, TIdx>;

    namespace trait
    {
        template<typename TDim, typename TIdx>
        struct AccToTag<alpaka::AccGpuCudaRt<TDim, TIdx>>
        {
            using type = alpaka::TagGpuCudaRt;
        };

        template<typename TDim, typename TIdx>
        struct TagToAcc<alpaka::TagGpuCudaRt, TDim, TIdx>
        {
            using type = alpaka::AccGpuCudaRt<TDim, TIdx>;
        };
    } // namespace trait
} // namespace alpaka

#endif // ALPAKA_ACC_GPU_CUDA_ENABLED
