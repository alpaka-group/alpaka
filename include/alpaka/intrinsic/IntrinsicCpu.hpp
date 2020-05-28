/* Copyright 2020 Sergei Bastrakov
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/intrinsic/Traits.hpp>

#include <bitset>

namespace alpaka
{
    namespace intrinsic
    {
        //#############################################################################
        //! The CPU intrinsic.
        class IntrinsicCpu : public concepts::Implements<ConceptIntrinsic, IntrinsicCpu>
        {
        public:
            //-----------------------------------------------------------------------------
            IntrinsicCpu() = default;
            //-----------------------------------------------------------------------------
            IntrinsicCpu(IntrinsicCpu const &) = delete;
            //-----------------------------------------------------------------------------
            IntrinsicCpu(IntrinsicCpu &&) = delete;
            //-----------------------------------------------------------------------------
            auto operator=(IntrinsicCpu const &) -> IntrinsicCpu & = delete;
            //-----------------------------------------------------------------------------
            auto operator=(IntrinsicCpu &&) -> IntrinsicCpu & = delete;
            //-----------------------------------------------------------------------------
            ~IntrinsicCpu() = default;
        };

        namespace traits
        {
            //#############################################################################
            template<>
            struct Popcount<
                IntrinsicCpu>
            {
                //-----------------------------------------------------------------------------
                static auto popcount(
                    intrinsic::IntrinsicCpu const & /*intrinsic*/,
                    unsigned int value)
                -> int
                {
#if BOOST_COMP_GNUC || BOOST_COMP_CLANG || BOOST_COMP_INTEL
                    return __builtin_popcount(value);
#elif BOOST_COMP_MSVC
                    return __popcnt(value);
#else
                    // Fallback to standard library
                    return static_cast<int>(std::bitset<32>(value).count());
#endif
                }

                //-----------------------------------------------------------------------------
                static auto popcount(
                    intrinsic::IntrinsicCpu const & /*intrinsic*/,
                    unsigned long long value)
                -> int
                {
#if BOOST_COMP_GNUC || BOOST_COMP_CLANG || BOOST_COMP_INTEL
                    return __builtin_popcountll(value);
#elif BOOST_COMP_MSVC
                    return static_cast<int>(__popcnt64(value));
#else
                    // Fallback to standard library
                    return static_cast<int>(std::bitset<64>(value).count());
#endif
                }
            };
        }
    }
}
