/* Copyright 2022 Sergei Bastrakov, Jeffrey Kelling, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/intrinsic/Traits.hpp"

namespace alpaka
{
    namespace detail
    {
        //! Fallback implementation of popcount.
        template<typename TValue>
        static auto popcountFallback(TValue value) -> std::int32_t
        {
            TValue count = 0;
            while(value != 0)
            {
                count += value & 1u;
                value >>= 1u;
            }
            return static_cast<std::int32_t>(count);
        }

        //! Fallback implementation of ffs.
        template<typename TValue>
        static auto ffsFallback(TValue value) -> std::int32_t
        {
            if(value == 0)
                return 0;
            std::int32_t result = 1;
            while((value & 1) == 0)
            {
                value >>= 1;
                result++;
            }
            return result;
        }

        template<typename TValue>
        static auto brevFallback(TValue value) -> TValue
        {
            if(value == 0 || value == ~TValue{0})
                return value;

            TValue mask{1};
            TValue result{0};

            const std::uint32_t bits = sizeof(value) * 8;
            for(std::uint32_t bit = 0; bit < bits; bit++)
            {
                if(value & mask)
                {
                    result |= mask << (bits - bit - 1);
                }
                mask = mask << 1;
            }

            return result;
        }

        template<typename TValue>
        static auto clzFallback(TValue value) -> std::int32_t
        {
            if(value == 0)
                return 32;
            else if(value == ~TValue{0})
                return 0;
            std::int32_t n = 0;

            TValue mask{1};

            const std::uint32_t bits = sizeof(value) * 8;
            for(std::uint32_t bit = 0; bit < bits; bit++)
            {
                if((value & mask) == 0)
                {
                    n += 1;
                }
                mask = mask << 1;
            }

            return n;
        }

    } // namespace detail

    //! The Fallback intrinsic.
    class IntrinsicFallback : public interface::Implements<ConceptIntrinsic, IntrinsicFallback>
    {
    };

    namespace trait
    {
        template<>
        struct Popcount<IntrinsicFallback>
        {
            static auto popcount(IntrinsicFallback const& /*intrinsic*/, std::uint32_t value) -> std::int32_t
            {
                return alpaka::detail::popcountFallback(value);
            }

            static auto popcount(IntrinsicFallback const& /*intrinsic*/, std::uint64_t value) -> std::int32_t
            {
                return alpaka::detail::popcountFallback(value);
            }
        };

        template<>
        struct Ffs<IntrinsicFallback>
        {
            static auto ffs(IntrinsicFallback const& /*intrinsic*/, std::int32_t value) -> std::int32_t
            {
                return alpaka::detail::ffsFallback(value);
            }

            static auto ffs(IntrinsicFallback const& /*intrinsic*/, std::int64_t value) -> std::int32_t
            {
                return alpaka::detail::ffsFallback(value);
            }
        };

        template<>
        struct Brev<IntrinsicFallback>
        {
            static auto brev(IntrinsicFallback const& /*intrinsic*/, std::uint32_t value) -> std::uint32_t
            {
                return alpaka::detail::brevFallback(value);
            }

            static auto brev(IntrinsicFallback const& /*intrinsic*/, std::uint64_t value) -> std::uint64_t
            {
                return alpaka::detail::brevFallback(value);
            }
        };

        template<>
        struct Clz<IntrinsicFallback>
        {
            static auto clz(IntrinsicFallback const& /*intrinsic*/, std::uint32_t value) -> std::int32_t
            {
                return alpaka::detail::clzFallback(value);
            }

            static auto clz(IntrinsicFallback const& /*intrinsic*/, std::uint64_t value) -> std::int32_t
            {
                return alpaka::detail::clzFallback(value);
            }
        };

    } // namespace trait
} // namespace alpaka
