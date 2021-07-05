/* Copyright 2021 Jiri Vyskocil
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>

#include <array>

namespace alpaka
{
    namespace rand
    {
        /// Get high 32 bits of a 64-bit number
        ALPAKA_FN_HOST_ACC constexpr static std::uint64_t hi32(std::uint64_t x)
        {
            return (x >> 32);
        }

        /// Get low 32 bits of a 64-bit number
        ALPAKA_FN_HOST_ACC constexpr static std::uint64_t lo32(std::uint64_t x)
        {
            return (x & 0xffffffff);
        }

        /** Multiply two 64-bit numbers and split the result into high and low 32 bits
         *
         * @param a first 64-bit multiplier
         * @param b second 64-bit multiplier
         * @return the product a*b split into an array of [high 32 bits, low 32 bits]
         */
        // TODO: See single-instruction implementations in original Philox source code
        ALPAKA_FN_HOST_ACC constexpr static std::array<std::uint32_t, 2> mulhilo32(std::uint64_t a, std::uint64_t b)
        {
            std::uint64_t res64 = a * b;
            return {static_cast<std::uint32_t>(hi32(res64)), static_cast<std::uint32_t>(lo32(res64))};
        }
    } // namespace rand
} // namespace alpaka
