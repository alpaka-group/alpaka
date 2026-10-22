/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

/** @file This file provides a basic implementation of a SIMD vector.
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
#include "alpaka/simd/concepts.hpp"
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
        /** Simd array storge for vector data
         *
         * The storage is aligned for native simd usage.
         */
        template<typename T_Type, uint32_t T_width>
        struct alignas(alpaka::internal::optimalAlignment<T_Type, T_width, Alignment<sizeof(T_Type) * T_width>>())
            EmuSimd : protected std::array<T_Type, T_width>
        {
            using BaseType = std::array<T_Type, T_width>;

            using value_type = typename BaseType::value_type;
            using reference = typename BaseType::reference;

            using BaseType::operator[];

            constexpr EmuSimd() = default;

            constexpr EmuSimd(EmuSimd const& other)
            {
                // attention:  using default constructor results in bad performance
                for(uint32_t i = 0u; i < T_width; ++i)
                    BaseType::operator[](i) = other[i];
            }

            constexpr EmuSimd(EmuSimd&&) = default;
            constexpr EmuSimd& operator=(EmuSimd&& rhs) = default;

            constexpr EmuSimd& operator=(EmuSimd const& rhs) = default;

            constexpr EmuSimd& operator=(T_Type const value)
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
            constexpr EmuSimd(T_Args&&... args) : BaseType{std::forward<T_Args>(args)...}
            {
            }

            constexpr EmuSimd(BaseType const& base) : BaseType{base}
            {
            }

            /** static cast the instance to the parent class
             *
             * This method is mostly used to get access to native arithmetic and comparison operators.
             * @{
             */
            constexpr auto& asNativeType()
            {
                return static_cast<EmuSimd&>(*this);
            }

            constexpr auto const& asNativeType() const
            {
                return static_cast<EmuSimd const&>(*this);
            }

            /** @} */

            static constexpr auto fill(T_Type value)
            {
                return EmuSimd([&value](uint32_t const) { return value; });
            }

            template<typename F>
            requires std::invocable<F, std::integral_constant<uint32_t, 0u>>
            constexpr explicit EmuSimd(F&& generator)
                : EmuSimd(std::forward<F>(generator), std::make_integer_sequence<uint32_t, T_width>{})
            {
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
                    *reinterpret_cast<std::remove_const_t<ALPAKA_TYPEOF(*this)>*>(data) = (*this);
                else
                {
                    for(uint32_t i = 0u; i < T_width; ++i)
                        data[i] = asNativeType()[i];
                }
            }

            template<alpaka::concepts::SimdMask T_Mask, alpaka::concepts::Simd T_Simd>
            friend struct SimdWhereExpr;

            /** element wise conditional value update where t is a scalar */
            constexpr void update(alpaka::concepts::SimdMask auto const& mask, alpaka::concepts::Simd auto const& t)
            {
                using MaskType = ALPAKA_TYPEOF(valueMaskCast<T_Type>(t[0]));
                if constexpr(std::same_as<MaskType, bool>)
                {
                    for(uint32_t i = 0u; i < T_width; ++i)
                        asNativeType()[i] = (mask[i] ? t[i] : asNativeType()[i]);
                }
                else
                {
                    for(uint32_t i = 0u; i < T_width; ++i)
                        asNativeType()[i] = std::bit_cast<T_Type>(
                            (mask.asNativeType()[i] & std::bit_cast<MaskType>(t[i]))
                            | (~mask.asNativeType()[i] & std::bit_cast<MaskType>(asNativeType()[i])));
                }
            }

            /** element wise conditional value update where t is a scalar */
            constexpr void update(alpaka::concepts::SimdMask auto const& mask, T_Type const& t)
            {
                using MaskType = ALPAKA_TYPEOF(valueMaskCast<T_Type>(t));
                if constexpr(std::same_as<MaskType, bool>)
                {
                    for(uint32_t i = 0u; i < T_width; ++i)
                        asNativeType()[i] = (mask[i] ? t : asNativeType()[i]);
                }
                else
                {
                    for(uint32_t i = 0u; i < T_width; ++i)
                        asNativeType()[i] = std::bit_cast<T_Type>(
                            (mask.asNativeType()[i] & std::bit_cast<MaskType>(t))
                            | (~mask.asNativeType()[i] & std::bit_cast<MaskType>(asNativeType()[i])));
                }
            }

            /** assign operator
             */
#define ALPAKA_VECTOR_ASSIGN_OP(op)                                                                                   \
    constexpr EmuSimd& operator op(EmuSimd const& rhs)                                                                \
    {                                                                                                                 \
        for(uint32_t i = 0u; i < T_width; i++)                                                                        \
        {                                                                                                             \
            asNativeType()[i] op rhs[i];                                                                              \
        }                                                                                                             \
        return *this;                                                                                                 \
    }                                                                                                                 \
    constexpr EmuSimd& operator op(T_Type const value)                                                                \
    {                                                                                                                 \
        for(uint32_t i = 0u; i < T_width; i++)                                                                        \
        {                                                                                                             \
            asNativeType()[i] op value;                                                                               \
        }                                                                                                             \
        return *this;                                                                                                 \
    }

            ALPAKA_VECTOR_ASSIGN_OP(+=)
            ALPAKA_VECTOR_ASSIGN_OP(-=)
            ALPAKA_VECTOR_ASSIGN_OP(/=)
            ALPAKA_VECTOR_ASSIGN_OP(*=)

#undef ALPAKA_VECTOR_ASSIGN_OP

        private:
            template<typename F, uint32_t... Is>
            constexpr explicit EmuSimd(F&& generator, std::integer_sequence<uint32_t, Is...>)
                : BaseType{generator(std::integral_constant<uint32_t, Is>{})...}
            {
            }
        };

#define ALPAKA_VECTOR_BINARY_OP(typenameOrConcept, op)                                                                \
    template<typenameOrConcept T_Type, uint32_t T_width>                                                              \
    constexpr auto operator op(const EmuSimd<T_Type, T_width>& lhs, const EmuSimd<T_Type, T_width>& rhs)              \
    {                                                                                                                 \
        EmuSimd<T_Type, T_width> ret{};                                                                               \
        for(uint32_t i = 0u; i < T_width; i++)                                                                        \
            ret[i] = lhs[i] op rhs[i];                                                                                \
        return ret;                                                                                                   \
    }                                                                                                                 \
    template<typenameOrConcept T_Type, uint32_t T_width>                                                              \
    constexpr auto operator op(const EmuSimd<T_Type, T_width>& lhs, T_Type rhs)                                       \
    {                                                                                                                 \
        EmuSimd<T_Type, T_width> ret{};                                                                               \
        for(uint32_t i = 0u; i < T_width; i++)                                                                        \
            ret[i] = lhs[i] op rhs;                                                                                   \
        return ret;                                                                                                   \
    }                                                                                                                 \
    template<typenameOrConcept T_Type, uint32_t T_width>                                                              \
    constexpr auto operator op(T_Type lhs, const EmuSimd<T_Type, T_width>& rhs)                                       \
    {                                                                                                                 \
        EmuSimd<T_Type, T_width> ret{};                                                                               \
        for(uint32_t i = 0u; i < T_width; i++)                                                                        \
            ret[i] = lhs op rhs[i];                                                                                   \
        return ret;                                                                                                   \
    }

        ALPAKA_VECTOR_BINARY_OP(typename, +)
        ALPAKA_VECTOR_BINARY_OP(typename, -)
        ALPAKA_VECTOR_BINARY_OP(typename, *)
        ALPAKA_VECTOR_BINARY_OP(typename, /)
        ALPAKA_VECTOR_BINARY_OP(std::integral, %)
        ALPAKA_VECTOR_BINARY_OP(std::integral, <<)
        ALPAKA_VECTOR_BINARY_OP(std::integral, >>)
        ALPAKA_VECTOR_BINARY_OP(std::integral, &)
        ALPAKA_VECTOR_BINARY_OP(std::integral, |)
        ALPAKA_VECTOR_BINARY_OP(std::integral, ^)

#undef ALPAKA_VECTOR_BINARY_OP

    } // namespace internal

    namespace trait
    {
        template<concepts::Api T_Api, typename T_Type, uint32_t T_width>
        struct GetSimdStorageType
        {
            using type = internal::EmuSimd<T_Type, T_width>;
        };
    } // namespace trait
} // namespace alpaka
