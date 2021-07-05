/* Copyright 2021 Jiri Vyskocil
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED) || defined(ALPAKA_ACC_GPU_CUDA_ENABLED)

#    include <functional>
#    include <initializer_list>
#    include <numeric>
#    include <type_traits>

/// Helper struct providing [] subscript access to CUDA uint4 type
struct uint4_array : public uint4
{
    using value_type = uint32_t;

    __host__ __device__ uint4_array(std::initializer_list<uint32_t> init)
    {
        auto it = std::begin(init);
        x = *it++;
        y = *it++;
        z = *it++;
        w = *it++;
    }
    __host__ __device__ constexpr uint32_t& operator[](int const k)
    {
        return k == 0 ? x : (k == 1 ? y : (k == 2 ? z : w));
    }

    __host__ __device__ const uint32_t& operator[](int const k) const
    {
        return k == 0 ? x : (k == 1 ? y : (k == 2 ? z : w));
    }
};

/// Helper struct providing [] subscript access to CUDA uint2 type
struct uint2_array : uint2
{
    using value_type = uint32_t;

    __host__ __device__ uint2_array(std::initializer_list<uint32_t> init)
    {
        auto it = std::begin(init);
        x = *it++;
        y = *it++;
    }
    __host__ __device__ constexpr uint32_t& operator[](int const k)
    {
        return k == 0 ? x : y;
    }
    __host__ __device__ const uint32_t& operator[](int const k) const
    {
        return k == 0 ? x : y;
    }
};

namespace std
{
    /// Specialization of std::tuple_size for \a uint4_array
    template<>
    struct tuple_size<uint4_array> : integral_constant<size_t, 4>
    {
    };
    /// Specialization of std::tuple_size for \a uint2_array
    template<>
    struct tuple_size<uint2_array> : integral_constant<size_t, 2>
    {
    };
} // namespace std

#endif
