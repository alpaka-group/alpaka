/* Copyright 2022 Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/intrinsic/IntrinsicFallback.hpp"
#include "alpaka/intrinsic/Traits.hpp"

#include <cstdint>

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <sycl/sycl.hpp>

namespace alpaka
{
    //! The SYCL intrinsic.
    class IntrinsicGenericSycl : public interface::Implements<ConceptIntrinsic, IntrinsicGenericSycl>
    {
    };
} // namespace alpaka

namespace alpaka::trait
{
    template<>
    struct Popcount<IntrinsicGenericSycl>
    {
        static auto popcount(IntrinsicGenericSycl const&, std::uint32_t value) -> std::int32_t
        {
            return static_cast<std::int32_t>(sycl::popcount(value));
        }

        static auto popcount(IntrinsicGenericSycl const&, std::uint64_t value) -> std::int32_t
        {
            return static_cast<std::int32_t>(sycl::popcount(value));
        }
    };

    template<>
    struct Ffs<IntrinsicGenericSycl>
    {
        static auto ffs(IntrinsicGenericSycl const&, std::int32_t value) -> std::int32_t
        {
            // There is no FFS operation in SYCL but we can emulate it using popcount.
            return (value == 0) ? 0 : sycl::popcount(value ^ ~(-value));
        }

        static auto ffs(IntrinsicGenericSycl const&, std::int64_t value) -> std::int32_t
        {
            // There is no FFS operation in SYCL but we can emulate it using popcount.
            return (value == 0l) ? 0 : static_cast<std::int32_t>(sycl::popcount(value ^ ~(-value)));
        }
    };

    template<>
    struct Brev<IntrinsicGenericSycl>
    {
        static auto brev(IntrinsicGenericSycl const&, std::uint32_t value) -> std::uint32_t
        {
	    value = ((value & 0xaaaaaaaa) >> 1) | ((value & 0x55555555) << 1);
            value = ((value & 0xcccccccc) >> 2) | ((value & 0x33333333) << 2);
            value = ((value & 0xf0f0f0f0) >> 4) | ((value & 0x0f0f0f0f) << 4);
            value = ((value & 0xff00ff00) >> 8) | ((value & 0x00ff00ff) << 8);
            return (value >> 16) | (value << 16);
        }

        static auto brev(IntrinsicGenericSycl const&, std::uint64_t value) -> std::uint64_t
        {
            value = ((value & 0xAAAAAAAAAAAAAAAAULL) >> 1)  | ((value & 0x5555555555555555ULL) << 1);
            value = ((value & 0xCCCCCCCCCCCCCCCCULL) >> 2)  | ((value & 0x3333333333333333ULL) << 2);
            value = ((value & 0xF0F0F0F0F0F0F0F0ULL) >> 4)  | ((value & 0x0F0F0F0F0F0F0F0FULL) << 4);
            value = ((value & 0xFF00FF00FF00FF00ULL) >> 8)  | ((value & 0x00FF00FF00FF00FFULL) << 8);
            value = ((value & 0xFFFF0000FFFF0000ULL) >> 16) | ((value & 0x0000FFFF0000FFFFULL) << 16);
            return (value >> 32) | (value << 32);
        }
    };

    template<>
    struct Clz<IntrinsicGenericSycl>
    {
        static auto clz(IntrinsicGenericSycl const&, std::uint32_t value) -> std::int32_t
        {
            return sycl::clz(value);
        }

        static auto clz(IntrinsicGenericSycl const&, std::uint64_t value) -> std::int32_t
        {
            return sycl::clz(value);
        }
    };

} // namespace alpaka::trait

#endif
