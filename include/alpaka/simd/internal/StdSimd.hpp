/* Copyright 2026 René Widera
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
#include "alpaka/mem/Alignment.hpp"
#include "alpaka/simd/concepts.hpp"
#include "alpaka/simd/simdConfig.hpp"
#include "alpaka/simd/trait.hpp"
#include "alpaka/vecConcepts.hpp"

#include <type_traits>

#if ALPAKA_HAS_STD_SIMD

namespace alpaka
{
    namespace internal
    {
        template<typename T_Type, uint32_t T_width>
        struct StdSimd
            : protected alpakaStdSimd::rebind_simd_t<T_Type, alpakaStdSimd::fixed_size_simd<T_Type, T_width>>
        {
            using BaseType = alpakaStdSimd::rebind_simd_t<T_Type, alpakaStdSimd::fixed_size_simd<T_Type, T_width>>;

            using value_type = typename BaseType::value_type;
            using reference = typename BaseType::reference;

            using BaseType::operator[];

            constexpr StdSimd() = default;
            constexpr StdSimd(StdSimd const&) = default;
            constexpr StdSimd(StdSimd&&) = default;
            constexpr StdSimd& operator=(StdSimd&& rhs) = default;

            constexpr StdSimd& operator=(StdSimd const& rhs) = default;

            constexpr StdSimd& operator=(T_Type const value)
            {
                this->asNativeType() = value;
                return *this;
            }

            // constructor is required because exposing the array constructors does not work
            template<typename... T_Args>
            requires(sizeof...(T_Args) == T_width && (std::same_as<T_Args, T_Type> && ...))
            ALPAKA_FN_HOST_ACC StdSimd(T_Args&&... args)
                : BaseType([=](int i) constexpr { return std::array<T_Type, T_width>{args...}[i]; })
            {
            }

            constexpr StdSimd(BaseType const& nativeSimd) : BaseType{nativeSimd}
            {
            }

            /** static cast the instance to the parent std::simd class
             *
             * This method is mostly used to get access to native arithmetic and comparison operators.
             * @{
             */
            constexpr auto& asNativeType()
            {
                return static_cast<BaseType&>(*this);
            }

            constexpr auto const& asNativeType() const
            {
                return static_cast<BaseType const&>(*this);
            }

            /** @} */

            constexpr decltype(auto) where(alpaka::concepts::SimdMask auto const& mask) const
            {
                return alpakaStdSimd::where(mask.asNativeType(), asNativeType());
            }

            constexpr decltype(auto) where(alpaka::concepts::SimdMask auto const& mask)
            {
                return alpakaStdSimd::where(mask.asNativeType(), asNativeType());
            }

            static constexpr auto fill(T_Type value)
            {
                return StdSimd{BaseType(value)};
            }

            constexpr void copyFrom(T_Type const* data, alpaka::concepts::Alignment auto alignment)
            {
                if constexpr((alignment.template get<T_Type>() % alpakaStdSimd::memory_alignment_v<BaseType>) == 0u)
                    this->asNativeType().copy_from(data, alpakaStdSimd::vector_aligned);
                else
                    this->asNativeType().copy_from(data, alpakaStdSimd::element_aligned);
            }

            constexpr void copyTo(auto* data, alpaka::concepts::Alignment auto alignment) const
            {
                if constexpr((alignment.template get<T_Type>() % alpakaStdSimd::memory_alignment_v<BaseType>) == 0u)
                    this->asNativeType().copy_to(data, alpakaStdSimd::vector_aligned);
                else
                    this->asNativeType().copy_to(data, alpakaStdSimd::element_aligned);
            }

            /** assign operator
             */
#    define ALPAKA_VECTOR_ASSIGN_OP(op)                                                                               \
        constexpr StdSimd& operator op(StdSimd const& rhs)                                                            \
        {                                                                                                             \
            this->asNativeType() op rhs.asNativeType();                                                               \
            return *this;                                                                                             \
        }                                                                                                             \
        constexpr StdSimd& operator op(T_Type const value)                                                            \
        {                                                                                                             \
            this->asNativeType() op value;                                                                            \
            return *this;                                                                                             \
        }

            ALPAKA_VECTOR_ASSIGN_OP(+=)
            ALPAKA_VECTOR_ASSIGN_OP(-=)
            ALPAKA_VECTOR_ASSIGN_OP(/=)
            ALPAKA_VECTOR_ASSIGN_OP(*=)

#    undef ALPAKA_VECTOR_ASSIGN_OP
        };

#    define ALPAKA_VECTOR_BINARY_OP(typenameOrConcept, op)                                                            \
        template<typenameOrConcept T_Type, uint32_t T_width>                                                          \
        constexpr auto operator op(const StdSimd<T_Type, T_width>& lhs, const StdSimd<T_Type, T_width>& rhs)          \
        {                                                                                                             \
            return StdSimd<T_Type, T_width>{lhs.asNativeType() op rhs.asNativeType()};                                \
        }                                                                                                             \
        template<typenameOrConcept T_Type, uint32_t T_width>                                                          \
        constexpr auto operator op(const StdSimd<T_Type, T_width>& lhs, T_Type rhs)                                   \
        {                                                                                                             \
            return StdSimd<T_Type, T_width>{lhs.asNativeType() op rhs};                                               \
        }                                                                                                             \
        template<typenameOrConcept T_Type, uint32_t T_width>                                                          \
        constexpr auto operator op(T_Type lhs, const StdSimd<T_Type, T_width>& rhs)                                   \
        {                                                                                                             \
            return StdSimd<T_Type, T_width>{lhs op rhs.asNativeType()};                                               \
        }

        ALPAKA_VECTOR_BINARY_OP(typename, +)
        ALPAKA_VECTOR_BINARY_OP(typename, -)
        ALPAKA_VECTOR_BINARY_OP(typename, *)
        ALPAKA_VECTOR_BINARY_OP(typename, /)
        ALPAKA_VECTOR_BINARY_OP(std::integral, <<)
        ALPAKA_VECTOR_BINARY_OP(std::integral, >>)
        ALPAKA_VECTOR_BINARY_OP(std::integral, &)
        ALPAKA_VECTOR_BINARY_OP(std::integral, |)
        ALPAKA_VECTOR_BINARY_OP(std::integral, ^)

        /** Workaround clang + glibc 12 issue with std::simd modulo operator
         *
         * /usr/lib/gcc/x86_64-linux-gnu/12/../../../../include/c++/12/experimental/bits/simd_x86.h:1492:51: error:
         * explicit qualification required to use member '_S_divides' from dependent base class 1492 |           return
         * _Base::_S_minus(__x, _S_multiplies(__y, _S_divides(__x, __y)));
         *
         * This workaround is executing the operation lane by lane which can break SIMD usage if the auto vectorizer is
         * not understanding the code.
         */
#    if defined(__clang__) && defined(__GLIBCXX__) && (!defined(_GLIBCXX_RELEASE) || _GLIBCXX_RELEASE == 12)
        template<std::integral T_Type, uint32_t T_width>
        constexpr auto operator%(const StdSimd<T_Type, T_width>& lhs, const StdSimd<T_Type, T_width>& rhs)
        {
            using BaseType = typename StdSimd<T_Type, T_width>::BaseType;
            return StdSimd<T_Type, T_width>(
                BaseType([&](int i) { return lhs.asNativeType()[i] % rhs.asNativeType()[i]; }));
        }

        template<std::integral T_Type, uint32_t T_width>
        constexpr auto operator%(StdSimd<T_Type, T_width> const& lhs, T_Type rhs)
        {
            using BaseType = typename StdSimd<T_Type, T_width>::BaseType;
            return StdSimd<T_Type, T_width>(BaseType([&](int i) { return lhs.asNativeType()[i] % rhs; }));
        }

        template<std::integral T_Type, uint32_t T_width>
        constexpr auto operator%(T_Type lhs, StdSimd<T_Type, T_width> const& rhs)
        {
            using BaseType = typename StdSimd<T_Type, T_width>::BaseType;
            return StdSimd<T_Type, T_width>(BaseType([&](int i) { return lhs % rhs.asNativeType()[i]; }));
        }
#    else
        ALPAKA_VECTOR_BINARY_OP(std::integral, %)
#    endif
#    undef ALPAKA_VECTOR_BINARY_OP

    } // namespace internal

    namespace trait
    {
        template<typename T_Type, uint32_t T_width>
        requires(
            std::has_single_bit(T_width) && std::has_single_bit(sizeof(T_Type))
            && alpakaStdSimd::fixed_size_simd<T_Type, T_width>::size() > 0)
        struct GetSimdStorageType<alpaka::api::Host, T_Type, T_width>
        {
            using type = internal::StdSimd<T_Type, T_width>;
        };

    } // namespace trait
} // namespace alpaka
#endif
