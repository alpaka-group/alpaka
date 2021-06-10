/* Copyright 2021 Jiri Vyskocil
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

//TODO: Decide whether we want to use the cuintArray types (and where to put them). If not, remove this.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#    include <alpaka/rand/Philox/helpers/cuintArray.hpp>
#endif

#include <functional>
#include <numeric>
#include <type_traits>

namespace alpaka
{
    namespace meta
    {
        // --- IsArrayOrVector ----------------------------------------
        /** Checks whether T is an array or a vector type
         *
         * @tparam T a type to check
         */
        template<typename T>
        struct IsArrayOrVector
        {
            enum
            {
                value = false
            };
        };

        /** Specialization of \a IsArrayOrVector for vector types
         *
         * @tparam T inner type held in the vector
         * @tparam A vector allocator
         */
        template<typename T, typename A>
        struct IsArrayOrVector<std::vector<T, A>>
        {
            enum
            {
                value = true
            };
        };

        /** Specialization of \a IsArrayOrVector for array types
         *
         * @tparam T inner type held in the array
         * @tparam N size of the array
         */
        template<typename T, std::size_t N>
        struct IsArrayOrVector<std::array<T, N>>
        {
            enum
            {
                value = true
            };
        };

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
        /// Specialization of \a IsArrayOrVector for CUDA uint4_array
        template<>
        struct IsArrayOrVector<uint4_array>
        {
            enum
            {
                value = true
            };
        };

        /// Specialization of \a IsArrayOrVector for CUDA uint2_array
        template<>
        struct IsArrayOrVector<uint2_array>
        {
            enum
            {
                value = true
            };
        };
#endif
    } // namespace meta
} // namespace alpaka
