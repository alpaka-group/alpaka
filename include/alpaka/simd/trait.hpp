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

#include "alpaka/api/api.hpp"
#include "alpaka/api/concepts/api.hpp"

#include <concepts>
#include <type_traits>

namespace alpaka
{
    namespace trait
    {
        /** Get the storage type for a SIMD pack */
        template<concepts::Api T_Api, typename T_Type, uint32_t T_width>
        struct GetSimdStorageType;

        /** Get the storage type for a SIMD mask pack */
        template<concepts::Api T_Api, typename T_Type, uint32_t T_width>
        struct GetSimdMaskStorageType;
    } // namespace trait
} // namespace alpaka
