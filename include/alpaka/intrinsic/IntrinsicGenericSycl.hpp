/* Copyright 2020 Jan Stephan
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/intrinsic/Traits.hpp>
#include <alpaka/intrinsic/IntrinsicFallback.hpp>

#include <CL/sycl.hpp>

namespace alpaka
{
    //#############################################################################
    //! The CPU intrinsic.
    class IntrinsicGenericSycl : public concepts::Implements<ConceptIntrinsic, IntrinsicGenericSycl>
    {
    public:
        //-----------------------------------------------------------------------------
        IntrinsicGenericSycl() = default;
        //-----------------------------------------------------------------------------
        IntrinsicGenericSycl(IntrinsicGenericSycl const &) = delete;
        //-----------------------------------------------------------------------------
        IntrinsicGenericSycl(IntrinsicGenericSycl &&) = delete;
        //-----------------------------------------------------------------------------
        auto operator=(IntrinsicGenericSycl const &) -> IntrinsicGenericSycl & = delete;
        //-----------------------------------------------------------------------------
        auto operator=(IntrinsicGenericSycl &&) -> IntrinsicGenericSycl & = delete;
        //-----------------------------------------------------------------------------
        ~IntrinsicGenericSycl() = default;
    };

    namespace traits
    {
        //#############################################################################
        template<>
        struct Popcount<
            IntrinsicGenericSycl>
        {
            //-----------------------------------------------------------------------------
            static auto popcount(
                IntrinsicGenericSycl const & /*intrinsic*/,
                std::uint32_t value)
            -> std::int32_t
            {
                return static_cast<std::int32_t>(cl::sycl::popcount(value));
            }

            //-----------------------------------------------------------------------------
            static auto popcount(
                IntrinsicGenericSycl const & /*intrinsic*/,
                std::uint64_t value)
            -> std::int32_t
            {
                return static_cast<std::int32_t>(cl::sycl::popcount(value));
            }
        };

        //#############################################################################
        template<>
        struct Ffs<
            IntrinsicGenericSycl>
        {
            //-----------------------------------------------------------------------------
            static auto ffs(
                IntrinsicGenericSycl const & /*intrinsic*/,
                std::int32_t value)
            -> std::int32_t
            {
                // There is no FFS operation in SYCL but we can emulate it using popcount.
                return (value == 0) ? 0 : cl::sycl::popcount(value ^ ~(-value));
            }

            //-----------------------------------------------------------------------------
            static auto ffs(
                IntrinsicGenericSycl const & /*intrinsic*/,
                std::int64_t value)
            -> std::int32_t
            {
                // There is no FFS operation in SYCL but we can emulate it using popcount.
                return (value == 0l) ? 0 : static_cast<std::int32_t>(cl::sycl::popcount(value ^ ~(-value)));
            }
        };
    }
}

