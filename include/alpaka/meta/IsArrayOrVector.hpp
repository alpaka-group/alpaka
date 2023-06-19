/* Copyright 2022 Jiri Vyskocil, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/meta/CudaVectorArrayWrapper.hpp"

#include <functional>
#include <numeric>
#include <type_traits>
#include <vector>

namespace alpaka::meta
{
    /** Checks whether T is an array or a vector type
     *
     * @tparam T a type to check
     */
    template<typename T>
    struct IsArrayOrVector : std::false_type
    {
    };

    /** Specialization of \a IsArrayOrVector for vector types
     *
     * @tparam T inner type held in the vector
     * @tparam A vector allocator
     */
    template<typename T, typename A>
    struct IsArrayOrVector<std::vector<T, A>> : std::true_type
    {
    };

    /** Specialization of \a IsArrayOrVector for plain arrays
     *
     * @tparam T inner type held in the array
     * @tparam N size of the array
     */
    template<typename T, std::size_t N>
    struct IsArrayOrVector<T[N]> : std::true_type
    {
    };

    /** Specialization of \a IsArrayOrVector for std::array
     *
     * @tparam T inner type held in the array
     * @tparam N size of the array
     */
    template<typename T, std::size_t N>
    struct IsArrayOrVector<std::array<T, N>> : std::true_type
    {
    };

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    /// Specialization of \a IsArrayOrVector for CUDA vector array wrapper
    template<typename T, unsigned N>
    struct IsArrayOrVector<CudaVectorArrayWrapper<T, N>> : std::true_type
    {
    };
#endif
} // namespace alpaka::meta
