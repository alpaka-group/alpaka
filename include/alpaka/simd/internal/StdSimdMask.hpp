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
#include "alpaka/simd/internal/StdSimd.hpp"
#include "alpaka/simd/simdConfig.hpp"
#include "alpaka/simd/trait.hpp"

#include <type_traits>

#if ALPAKA_HAS_STD_SIMD

namespace alpaka
{
    namespace internal
    {
        template<typename T_Type, uint32_t T_width>
        struct StdSimdMask
            : protected alpakaStdSimd::
                  rebind_simd_t<T_Type, alpakaStdSimd::simd_mask<T_Type, alpakaStdSimd::simd_abi::fixed_size<T_width>>>
        {
            using BaseType = alpakaStdSimd::
                rebind_simd_t<T_Type, alpakaStdSimd::simd_mask<T_Type, alpakaStdSimd::simd_abi::fixed_size<T_width>>>;

            using value_type = typename BaseType::value_type;
            using reference = typename BaseType::reference;

            using BaseType::operator[];

            constexpr StdSimdMask() = default;
            constexpr StdSimdMask(StdSimdMask const&) = default;
            constexpr StdSimdMask(StdSimdMask&&) = default;
            constexpr StdSimdMask& operator=(StdSimdMask&& rhs) = default;

            constexpr StdSimdMask& operator=(StdSimdMask const& rhs) = default;

            constexpr StdSimdMask& operator=(T_Type const value)
            {
                this->asNativeType() = value;
                return *this;
            }

            // constructor is required because exposing the array constructors does not work
            template<typename... T_Args>
            requires(sizeof...(T_Args) == T_width && (std::same_as<T_Args, T_Type> && ...))
            constexpr StdSimdMask(T_Args const&... args) : BaseType{}
            {
                std::array<T_Type, T_width> const initArgs{ALPAKA_FORWARD(args)...};
                for(uint32_t i = 0u; i < T_width; ++i)
                    this->asNativeType()[i] = static_cast<bool>(initArgs[i]);
            }

            template<typename... T_Args>
            requires(sizeof...(T_Args) == T_width && (std::same_as<T_Args, bool> && ...))
            constexpr StdSimdMask(T_Args... args) : BaseType{}
            {
                std::array<bool, T_width> const initArgs{args...};
                for(uint32_t i = 0u; i < T_width; ++i)
                    this->asNativeType()[i] = initArgs[i];
            }

            constexpr StdSimdMask(BaseType const& nativeSimd) : BaseType{nativeSimd}
            {
            }

            /** static cast the instance to the parent std::simd_mask class
             *
             * This method is mostly used to get access to native comparison operators.
             *
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

            static constexpr auto fill(bool value)
            {
                return StdSimdMask{BaseType(value)};
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
        template<typename T_OtherStorage>                                                                             \
        constexpr StdSimdMask& operator op(StdSimdMask const& rhs)                                                    \
        {                                                                                                             \
            this->asNativeType() op rhs.asNativeType();                                                               \
            return *this;                                                                                             \
        }                                                                                                             \
        constexpr StdSimdMask& operator op(T_Type const value)                                                        \
        {                                                                                                             \
            this->asNativeType() op value;                                                                            \
            return *this;                                                                                             \
        }

            ALPAKA_VECTOR_ASSIGN_OP(&=)
            ALPAKA_VECTOR_ASSIGN_OP(|=)
            ALPAKA_VECTOR_ASSIGN_OP(^=)

#    undef ALPAKA_VECTOR_ASSIGN_OP
        };

#    define ALPAKA_VECTOR_BINARY_CMP_OP(returnSimdType, argSimdType, typenameOrConcept, op)                           \
        template<typenameOrConcept T_Type, uint32_t T_width>                                                          \
        constexpr auto operator op(const argSimdType<T_Type, T_width>& lhs, const argSimdType<T_Type, T_width>& rhs)  \
        {                                                                                                             \
            return returnSimdType<T_Type, T_width>{lhs.asNativeType() op rhs.asNativeType()};                         \
        }                                                                                                             \
        template<typenameOrConcept T_Type, uint32_t T_width>                                                          \
        constexpr auto operator op(const argSimdType<T_Type, T_width>& lhs, T_Type rhs)                               \
        {                                                                                                             \
            return returnSimdType<T_Type, T_width>{lhs.asNativeType() op rhs};                                        \
        }                                                                                                             \
        template<typenameOrConcept T_Type, uint32_t T_width>                                                          \
        constexpr auto operator op(T_Type lhs, const argSimdType<T_Type, T_width>& rhs)                               \
        {                                                                                                             \
            return returnSimdType<T_Type, T_width>{lhs op rhs.asNativeType()};                                        \
        }

        ALPAKA_VECTOR_BINARY_CMP_OP(StdSimdMask, StdSimd, typename, >=)
        ALPAKA_VECTOR_BINARY_CMP_OP(StdSimdMask, StdSimd, typename, >)
        ALPAKA_VECTOR_BINARY_CMP_OP(StdSimdMask, StdSimd, typename, <=)
        ALPAKA_VECTOR_BINARY_CMP_OP(StdSimdMask, StdSimd, typename, <)
        ALPAKA_VECTOR_BINARY_CMP_OP(StdSimdMask, StdSimd, typename, ==)
        ALPAKA_VECTOR_BINARY_CMP_OP(StdSimdMask, StdSimd, typename, !=)

        ALPAKA_VECTOR_BINARY_CMP_OP(StdSimdMask, StdSimdMask, typename, ==)
        ALPAKA_VECTOR_BINARY_CMP_OP(StdSimdMask, StdSimdMask, typename, !=)
        ALPAKA_VECTOR_BINARY_CMP_OP(StdSimdMask, StdSimdMask, typename, &&)
        ALPAKA_VECTOR_BINARY_CMP_OP(StdSimdMask, StdSimdMask, typename, ||)

#    undef ALPAKA_VECTOR_BINARY_CMP_OP

    } // namespace internal

    namespace trait
    {
        template<typename T_Type, uint32_t T_width>
        requires(
            std::has_single_bit(T_width) && std::has_single_bit(sizeof(T_Type))
            && alpakaStdSimd::fixed_size_simd_mask<T_Type, T_width>::size() > 0u)
        struct GetSimdMaskStorageType<alpaka::api::Host, T_Type, T_width>
        {
            using type = internal::StdSimdMask<T_Type, T_width>;
        };
    } // namespace trait
} // namespace alpaka
#endif
