/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

/** @file
 * @brief This file provides a basic implementation of a SIMD vector.
 *
 * The implementation is based on the class Vec:
 *   - the storge policy should become the native SIMD implementation e.g. std::simd
 *   - load/ store and simd specifis should be implemented in the storage policy
 *   - the name of storage policy should be changed
 *
 *   The current operator operations relay on compilers auto vectorization.
 */

#pragma once

#include "alpaka/api/api.hpp"
#include "alpaka/simd/internal/EmuSimd.hpp"
#include "alpaka/simd/internal/SmartMaskValueRef.hpp"
#include "alpaka/simd/internal/alignment.hpp"
#include "alpaka/simd/internal/utility.hpp"
#include "alpaka/simd/trait.hpp"

#include <concepts>
#include <type_traits>

namespace alpaka
{
    namespace internal
    {
        template<typename T_Type, uint32_t T_width>
        struct alignas(alpaka::internal::optimalAlignment<
                       ALPAKA_TYPEOF(internal::valueMaskCast<T_Type>(true)),
                       T_width,
                       Alignment<sizeof(ALPAKA_TYPEOF(internal::valueMaskCast<T_Type>(true))) * T_width>>())
            EmuSimdMask : protected std::array<ALPAKA_TYPEOF(internal::valueMaskCast<T_Type>(true)), T_width>
        {
            using ValueMaskType = ALPAKA_TYPEOF(internal::valueMaskCast<T_Type>(true));

            using BaseType = std::array<ValueMaskType, T_width>;

            using value_type = bool;
            using reference = SmartMaskValueRef<bool, ValueMaskType>;

            using BaseType::operator[];

            constexpr reference operator[](std::integral auto const idx)
            {
                return reference(BaseType::operator[](idx));
            }

            constexpr EmuSimdMask() = default;

            constexpr EmuSimdMask(EmuSimdMask const& other)
            {
                // attention:  using default constructor results in bad performance
                for(uint32_t i = 0u; i < T_width; ++i)
                    BaseType::operator[](i) = other[i];
            }

            constexpr EmuSimdMask(EmuSimdMask&&) = default;

            constexpr EmuSimdMask& operator=(EmuSimdMask&& rhs) = default;

            constexpr EmuSimdMask& operator=(EmuSimdMask const& rhs) = default;

            constexpr EmuSimdMask& operator=(T_Type const value)
            {
                for(uint32_t i = 0u; i < T_width; i++)
                {
                    asNativeType()[i] = value;
                }
                return *this;
            }

            // constructor is required because exposing the array constructors does not work
            template<typename... T_Args>
            requires(sizeof...(T_Args) == T_width && (std::same_as<T_Args, T_Type> && ...))
            constexpr EmuSimdMask(T_Args const&... args) : BaseType{args...}
            {
            }

            template<typename... T_Args>
            requires(sizeof...(T_Args) == T_width && (std::same_as<T_Args, bool> && ...))
            constexpr EmuSimdMask(T_Args... args) : BaseType{valueMaskCast<T_Type>(args)...}
            {
            }

            constexpr EmuSimdMask(BaseType const& base) : BaseType{base}
            {
            }

            /** static cast the instance to the parent class
             *
             * This method is mostly used to get access to native comparison operators.
             * @{
             */
            constexpr auto& asNativeType()
            {
                return static_cast<EmuSimdMask&>(*this);
            }

            constexpr auto const& asNativeType() const
            {
                return static_cast<EmuSimdMask const&>(*this);
            }

            /** @} */

            static constexpr auto fill(bool value)
            {
                auto maskValue = valueMaskCast<T_Type>(value);
                BaseType ret{};
                for(uint32_t i = 0u; i < T_width; ++i)
                    ret[i] = maskValue;

                return EmuSimdMask(ret);
            }

            constexpr void copyFrom(T_Type const* data, alpaka::concepts::Alignment auto alignment)
            {
                if constexpr((alignment.template get<T_Type>() % alignof(ALPAKA_TYPEOF(*this))) == 0u)
                    *(this) = *reinterpret_cast<ALPAKA_TYPEOF(*this) const*>(data);
                else
                {
                    for(uint32_t i = 0u; i < T_width; ++i)
                        asNativeType()[i] = data[i];
                }
            }

            constexpr void copyTo(auto* data, alpaka::concepts::Alignment auto alignment) const
            {
                if constexpr((alignment.template get<T_Type>() % alignof(ALPAKA_TYPEOF(*this))) == 0u)
                    *reinterpret_cast<ALPAKA_TYPEOF(*this) const*>(data) = (*this);
                else
                {
                    for(uint32_t i = 0u; i < T_width; ++i)
                        data[i] = asNativeType()[i];
                }
            }

            /** assign operator
             */
#define ALPAKA_VECTOR_ASSIGN_OP(op)                                                                                   \
    template<typename T_OtherStorage>                                                                                 \
    constexpr EmuSimdMask& operator op(EmuSimdMask const& rhs)                                                        \
    {                                                                                                                 \
        for(uint32_t i = 0u; i < T_width; i++)                                                                        \
        {                                                                                                             \
            asNativeType()[i] op rhs[i];                                                                              \
        }                                                                                                             \
        return *this;                                                                                                 \
    }                                                                                                                 \
    constexpr EmuSimdMask& operator op(T_Type const value)                                                            \
    {                                                                                                                 \
        for(uint32_t i = 0u; i < T_width; i++)                                                                        \
        {                                                                                                             \
            asNativeType()[i] op value;                                                                               \
        }                                                                                                             \
        return *this;                                                                                                 \
    }
            ALPAKA_VECTOR_ASSIGN_OP(&=)
            ALPAKA_VECTOR_ASSIGN_OP(|=)
            ALPAKA_VECTOR_ASSIGN_OP(^=)

#undef ALPAKA_VECTOR_ASSIGN_OP
        };

#define ALPAKA_VECTOR_BINARY_CMP_OP(returnSimdType, argSimdType, typenameOrConcept, op)                               \
    template<typenameOrConcept T_Type, uint32_t T_width>                                                              \
    constexpr auto operator op(const argSimdType<T_Type, T_width>& lhs, const argSimdType<T_Type, T_width>& rhs)      \
    {                                                                                                                 \
        returnSimdType<T_Type, T_width> ret{};                                                                        \
        for(uint32_t i = 0u; i < T_width; i++)                                                                        \
            ret[i] = valueMaskCast<T_Type>(lhs[i] op rhs[i]);                                                         \
        return ret;                                                                                                   \
    }                                                                                                                 \
    template<typenameOrConcept T_Type, uint32_t T_width>                                                              \
    constexpr auto operator op(const argSimdType<T_Type, T_width>& lhs, T_Type rhs)                                   \
    {                                                                                                                 \
        returnSimdType<T_Type, T_width> ret{};                                                                        \
        for(uint32_t i = 0u; i < T_width; i++)                                                                        \
            ret[i] = valueMaskCast<T_Type>(lhs[i] op rhs);                                                            \
        return ret;                                                                                                   \
    }                                                                                                                 \
    template<typenameOrConcept T_Type, uint32_t T_width>                                                              \
    constexpr auto operator op(T_Type lhs, const argSimdType<T_Type, T_width>& rhs)                                   \
    {                                                                                                                 \
        returnSimdType<T_Type, T_width> ret{};                                                                        \
        for(uint32_t i = 0u; i < T_width; i++)                                                                        \
            ret[i] = valueMaskCast<T_Type>(lhs op rhs[i]);                                                            \
        return ret;                                                                                                   \
    }

        ALPAKA_VECTOR_BINARY_CMP_OP(EmuSimdMask, EmuSimd, typename, >=)
        ALPAKA_VECTOR_BINARY_CMP_OP(EmuSimdMask, EmuSimd, typename, >)
        ALPAKA_VECTOR_BINARY_CMP_OP(EmuSimdMask, EmuSimd, typename, <=)
        ALPAKA_VECTOR_BINARY_CMP_OP(EmuSimdMask, EmuSimd, typename, <)
        ALPAKA_VECTOR_BINARY_CMP_OP(EmuSimdMask, EmuSimd, typename, ==)
        ALPAKA_VECTOR_BINARY_CMP_OP(EmuSimdMask, EmuSimd, typename, !=)

        ALPAKA_VECTOR_BINARY_CMP_OP(EmuSimdMask, EmuSimdMask, typename, ==)
        ALPAKA_VECTOR_BINARY_CMP_OP(EmuSimdMask, EmuSimdMask, typename, !=)
        ALPAKA_VECTOR_BINARY_CMP_OP(EmuSimdMask, EmuSimdMask, typename, &&)
        ALPAKA_VECTOR_BINARY_CMP_OP(EmuSimdMask, EmuSimdMask, typename, ||)

#undef ALPAKA_VECTOR_BINARY_CMP_OP
    } // namespace internal

    namespace trait
    {
        template<concepts::Api T_Api, typename T_Type, uint32_t T_width>
        struct GetSimdMaskStorageType
        {
            using type = internal::EmuSimdMask<T_Type, T_width>;
        };
    } // namespace trait
} // namespace alpaka
