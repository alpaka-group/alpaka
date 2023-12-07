/*
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 *
 * SPDX-FileContributor: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/ApiHipRt.hpp"
#include "alpaka/mem/buf/BufUniformCudaHipRt.hpp"

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

namespace alpaka
{
    template<typename TElem, typename TDim, typename TIdx>
    using BufHipRt = BufUniformCudaHipRt<ApiHipRt, TElem, TDim, TIdx>;
} // namespace alpaka

#endif // ALPAKA_ACC_GPU_HIP_ENABLED
