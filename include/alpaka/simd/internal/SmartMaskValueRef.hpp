/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/simd/internal/utility.hpp"

#include <type_traits>

namespace alpaka::internal
{

    /** Simd mask reference
     *
     * A SIMD mask is not required to store its values as bool, it can store the values as a representable value type
     * where all bits are 1 for true and zero for false. To be able to assign values to a SIMD mask we can not return a
     * reference to the stored value because we need to cast the value during the write. For the read we need to cast
     * the value to bool.
     */
    template<typename T, typename T_ValueMask>
    struct SmartMaskValueRef
    {
        using value_type = T;
        using ValueMaskType = T_ValueMask;

        constexpr SmartMaskValueRef(ValueMaskType& ref) noexcept : valueRef(ref)
        {
        }

        // Convert to bool
        constexpr operator bool() const noexcept
        {
            if constexpr(std::is_same_v<ValueMaskType, bool>)
            {
                return valueRef;
            }
            else
            {
                return valueRef != ValueMaskType{0};
            }
        }

        // Optional: convert to raw storage type
        constexpr ValueMaskType value() const noexcept
        {
            return valueRef;
        }

        // Unary operators
        constexpr bool operator!() const noexcept
        {
            return !static_cast<bool>(*this);
        }

        constexpr ValueMaskType operator~() const noexcept
        {
            if constexpr(std::is_same_v<ValueMaskType, bool>)
                return !valueRef;
            else
                return ~valueRef;
        }

        // Binary operators returning values (not references)
        constexpr ValueMaskType operator|(SmartMaskValueRef const& rhs) const noexcept
        {
            return valueRef | rhs.valueRef;
        }

        constexpr ValueMaskType operator&(SmartMaskValueRef const& rhs) const noexcept
        {
            return valueRef & rhs.valueRef;
        }

        constexpr ValueMaskType operator^(SmartMaskValueRef const& rhs) const noexcept
        {
            if constexpr(std::is_same_v<ValueMaskType, bool>)
                return static_cast<bool>(*this) != static_cast<bool>(rhs);
            else
                return valueRef ^ rhs.valueRef;
        }

        // Comparison operators
        constexpr bool operator==(SmartMaskValueRef const& rhs) const noexcept
        {
            return static_cast<bool>(*this) == static_cast<bool>(rhs);
        }

        constexpr bool operator!=(SmartMaskValueRef const& rhs) const noexcept
        {
            return !(*this == rhs);
        }

#define SIMD_MASK_REF_ASSIGN_OP(OP, BOOL_FALLBACK)                                                                    \
    constexpr SmartMaskValueRef& operator OP(bool b) noexcept requires(!std::is_same_v<ValueMaskType, bool>)          \
    {                                                                                                                 \
        return (*this OP internal::valueMaskCast<ValueMaskType>(b));                                                  \
    }                                                                                                                 \
                                                                                                                      \
    constexpr SmartMaskValueRef& operator OP(ValueMaskType v) noexcept                                                \
    {                                                                                                                 \
        if constexpr(std::is_same_v<ValueMaskType, bool>)                                                             \
        {                                                                                                             \
            valueRef = valueRef BOOL_FALLBACK static_cast<bool>(v);                                                   \
        }                                                                                                             \
        else                                                                                                          \
        {                                                                                                             \
            valueRef OP v;                                                                                            \
        }                                                                                                             \
        return *this;                                                                                                 \
    }                                                                                                                 \
    static_assert(true)

        SIMD_MASK_REF_ASSIGN_OP(|=, ||);
        SIMD_MASK_REF_ASSIGN_OP(&=, &&);
        SIMD_MASK_REF_ASSIGN_OP(^=, !=);
        SIMD_MASK_REF_ASSIGN_OP(=, =);

#undef SIMD_MASK_REF_ASSIGN_OP

    private:
        ValueMaskType& valueRef;
    };

} // namespace alpaka::internal
