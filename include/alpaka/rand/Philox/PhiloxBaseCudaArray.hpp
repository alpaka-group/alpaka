/* Copyright 2022 Jiri Vyskocil
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/meta/CudaVectorArrayWrapper.hpp>

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED) || defined(ALPAKA_ACC_GPU_CUDA_ENABLED)

namespace alpaka::rand::engine
{
    namespace trait
    {
        template<typename TScalar>
        struct PhiloxResultContainerTraits;

        template<>
        struct PhiloxResultContainerTraits<float>
        {
            using type = meta::CudaVectorArrayWrapper<float, 4>;
        };

        template<>
        struct PhiloxResultContainerTraits<double>
        {
            using type = meta::CudaVectorArrayWrapper<double, 4>;
        };

        template<>
        struct PhiloxResultContainerTraits<int>
        {
            using type = meta::CudaVectorArrayWrapper<int, 4>;
        };

        template<>
        struct PhiloxResultContainerTraits<unsigned>
        {
            using type = meta::CudaVectorArrayWrapper<unsigned, 4>;
        };

        template<typename TScalar>
        using PhiloxResultContainer = typename PhiloxResultContainerTraits<TScalar>::type;
    } // namespace trait

    /** Philox backend using array-like interface to CUDA uintN types for the storage of Key and Counter
     *
     * @tparam TParams Philox algorithm parameters \sa PhiloxParams
     * @tparam TImpl engine type implementation (CRTP)
     */
    template<typename TParams, typename TImpl>
    class PhiloxBaseCudaArray
    {
        static_assert(TParams::counterSize == 4, "GPU Philox implemented only for counters of width == 4");

    public:
        using Counter
            = meta::CudaVectorArrayWrapper<unsigned, 4>; ///< Counter type = array-like interface to CUDA uint4
        using Key = meta::CudaVectorArrayWrapper<unsigned, 2>; ///< Key type = array-like interface to CUDA uint2
        template<typename TDistributionResultScalar>
        using ResultContainer = trait::PhiloxResultContainer<TDistributionResultScalar>; ///< Vector template for
                                                                                         ///< distribution results
    };
} // namespace alpaka::rand::engine

#endif
