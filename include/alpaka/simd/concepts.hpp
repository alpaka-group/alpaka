/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

/** @file
 * @brief This file provides a basic implementation of a SIMD vector.
 *
 * The implementation is based on the class Vec:
 *   - the storage policy should become the native SIMD implementation e.g. std::simd
 *   - load/ store and simd specifics should be implemented in the storage policy
 *   - the name of storage policy should be changed
 *
 *   The current operator operations rely on compilers auto vectorization.
 */

#pragma once

#include <concepts>
#include <cstdint>
#include <type_traits>

namespace alpaka
{
    namespace trait
    {
        template<typename T>
        struct IsSimd : std::false_type
        {
        };

        template<typename T>
        struct IsSimdMask : std::false_type
        {
        };

    } // namespace trait

    template<typename T>
    constexpr bool isSimd_v = trait::IsSimd<T>::value;

    template<typename T>
    constexpr bool isSimdMask_v = trait::IsSimdMask<T>::value;

    namespace concepts
    {
        template<typename T>
        concept Simd = isSimd_v<T>;

        template<typename T>
        concept SimdMask = isSimdMask_v<T>;

        template<typename T>
        concept SimdOrScalar = (isSimd_v<T> || std::integral<T> || std::floating_point<T>);

        template<typename T, typename T_RequiredComponent>
        concept TypeOrSimd = (isSimd_v<T> || std::is_same_v<T, T_RequiredComponent>);

        template<typename T, typename T_RequiredComponent>
        concept SimdOrConvertibleType = (isSimd_v<T> || std::is_convertible_v<T, T_RequiredComponent>);
    } // namespace concepts
} // namespace alpaka
