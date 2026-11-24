/* Copyright 2022 Jiri Vyskocil, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/rand/engine/philox/multiplyAndSplit64to32.hpp"

#include <cstdint>
#include <utility>

namespace alpaka::rand::engine::internal
{
    /** Constants used in the Philox algorithm
     *
     * The numbers are taken from the reference Philox implementation:
     *
     * J. K. Salmon, M. A. Moraes, R. O. Dror and D. E. Shaw, "Parallel random numbers: As easy as 1, 2, 3,"
     * SC '11: Proceedings of 2011 International Conference for High Performance Computing, Networking,
     * Storage and Analysis, 2011, pp. 1-12, doi: 10.1145/2063384.2063405.
     *
     * static const data members are transformed into functions, because GCC
     * assumes types with static data members to be not mappable and makes not
     * exception for constexpr ones. This is a valid interpretation of the
     * OpenMP <= 4.5 standard. In OpenMP >= 5.0 types with any kind of static
     * data member are mappable.
     */
    class PhiloxConstants
    {
    public:
        /// First Weyl sequence parameter: the golden ratio
        static consteval std::uint64_t weyl64p0()
        {
            return 0x9E37'79B9'7F4A'7C15;
        }

        /// Second Weyl sequence parameter: \f$ \sqrt{3}-1 \f$
        static consteval std::uint64_t weyl64p1()
        {
            return 0xBB67'AE85'84CA'A73B;
        }

        /// 1st Weyl sequence parameter, 32 bits
        static consteval std::uint32_t weyl32p0()
        {
            return high32Bits(weyl64p0());
        }

        /// 2nd Weyl sequence parameter, 32 bits
        static consteval std::uint32_t weyl32p1()
        {
            return high32Bits(weyl64p1());
        }

        /// First Philox S-box multiplier, applied to counter word 0 (Random123 PHILOX_M4x32_0)
        static consteval std::uint32_t multipliter4x32p0()
        {
            return 0xD251'1F53;
        }

        /// Second Philox S-box multiplier, applied to counter word 2 (Random123 PHILOX_M4x32_1)
        static consteval std::uint32_t multipliter4x32p1()
        {
            return 0xCD9E'8D57;
        }
    };
} // namespace alpaka::rand::engine::internal
