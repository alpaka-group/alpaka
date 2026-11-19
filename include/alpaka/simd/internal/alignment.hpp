/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

/** @file
 * @brief This file provides a basic implementation of a SIMD vector.
 *
 * The implementation is based on the class Vec:
 *   - the storge policy should become the native SIMD implementation e.g. std::simd
 *   - load/ store and simd specifis should be implemented in the storage policy
 *   - the name of storage policy should be changed
 *
 *   The current operator operations relay on compilers auto vectorization.
 */

#pragma once

#include "alpaka/mem/Alignment.hpp"

#include <bit>
#include <cstdint>
#include <functional>

namespace alpaka::internal
{
    /** Calculates the best alignment based
     *
     * Takes care that the alignment never exceeds T_Alignment.
     * In the worst case the alignment falls back to the alignment of the component type.
     */
    template<typename T_ValueType, uint32_t T_numElements, alpaka::concepts::Alignment T_Alignment>
    consteval uint32_t optimalAlignment()
    {
        constexpr uint32_t currentTypeAlignment = static_cast<uint32_t>(alignof(T_ValueType));
        if constexpr(T_numElements % 2 != 0u)
            return currentTypeAlignment;

        constexpr uint32_t dataSizeInBytes = static_cast<uint32_t>(sizeof(T_ValueType) * T_numElements);
        constexpr uint32_t alignment = std::min(T_Alignment::template get<T_ValueType>(), dataSizeInBytes);
        if constexpr(std::has_single_bit(alignment))
            return alignment;

        return static_cast<uint32_t>(alignof(T_ValueType));
    }
} // namespace alpaka::internal
