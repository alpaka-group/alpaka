/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

/** @file
 * @brief This file provides a basic implementation of a SIMD vector.
 *
 * The implementation is based on the class Vec:
 *   - the storge policy should become the native SIMD implementation e.g. std::simd
 *   - load/ store and simd specifics should be implemented in the storage policy
 *   - the name of storage policy should be changed
 *
 *   The current operator operations rely on compilers auto vectorization.
 */

#pragma once

#include "alpaka/Simd.hpp"

namespace alpaka
{
    template<concepts::SimdMask T_Mask, concepts::Simd T_Simd>
    struct SimdWhereExpr
    {
        T_Mask const& m_mask;
        T_Simd& value;

        constexpr SimdWhereExpr(T_Mask const& m, T_Simd& v) : m_mask(m), value(v)
        {
        }

        // disable copy and move constructors/operators to avoid pointing to invalid references.
        constexpr SimdWhereExpr(SimdWhereExpr const&) = delete;
        constexpr SimdWhereExpr(SimdWhereExpr&&) = delete;
        constexpr SimdWhereExpr& operator=(SimdWhereExpr const&) = delete;
        constexpr SimdWhereExpr& operator=(SimdWhereExpr&&) = delete;

        using value_type = typename T_Simd::value_type;

        constexpr void operator=(concepts::Simd auto const& rhs)
            requires std::same_as<value_type, typename ALPAKA_TYPEOF(rhs)::value_type>
        {
            if constexpr(requires { value.where(m_mask); })
                value.where(m_mask) = rhs.asNativeType();
            else
                value.update(m_mask, rhs);
        }

        constexpr void operator=(concepts::LosslesslyConvertible<value_type> auto const& rhs)
        {
            if constexpr(requires { value.where(m_mask); })
                value.where(m_mask) = rhs;
            else
                value.update(m_mask, static_cast<value_type>(rhs));
        }

#define ALPAKA_SIMD_EXPR_ASSIGN_OP(op_name, op)                                                                       \
    constexpr void operator op_name(concepts::Simd auto const& rhs)                                                   \
    {                                                                                                                 \
        if constexpr(requires { value.where(m_mask); })                                                               \
            value.where(m_mask) op_name rhs.asNativeType();                                                           \
        else                                                                                                          \
            value.update(m_mask, value op rhs);                                                                       \
    }                                                                                                                 \
    constexpr void operator op_name(concepts::LosslesslyConvertible<value_type> auto const& rhs)                      \
    {                                                                                                                 \
        if constexpr(requires { value.where(m_mask); })                                                               \
            value.where(m_mask) op_name rhs;                                                                          \
        else                                                                                                          \
            value.update(m_mask, value op rhs);                                                                       \
    }

        ALPAKA_SIMD_EXPR_ASSIGN_OP(+=, +)
        ALPAKA_SIMD_EXPR_ASSIGN_OP(-=, -)
        ALPAKA_SIMD_EXPR_ASSIGN_OP(/=, /)
        ALPAKA_SIMD_EXPR_ASSIGN_OP(*=, *)

#undef ALPAKA_SIMD_EXPR_ASSIGN_OP
    };

    /** Conditionally update each component of an SIMD pack
     *
     * @param mask SIMD pack of booleans, where each component is true for the element in v which should be overwritten
     * with the value assigned to the returned expression
     * @param value SIMD vector to which the mask is applied
     */
    template<concepts::SimdMask T_Mask, concepts::Simd T_Simd>
    constexpr SimdWhereExpr<T_Mask, T_Simd> where(T_Mask const& mask, T_Simd& value)
    {
        return {mask, value};
    }
} // namespace alpaka
