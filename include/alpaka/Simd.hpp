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

#include "alpaka/SimdMask.hpp"
#include "alpaka/Vec.hpp"
#include "alpaka/cast.hpp"
#include "alpaka/core/util.hpp"
#include "alpaka/mem/Alignment.hpp"
#include "alpaka/simd/concepts.hpp"
#include "alpaka/simd/trait.hpp"
#include "alpaka/trait.hpp"
#include "simd/internal/EmuSimd.hpp"
#include "simd/internal/StdSimd.hpp"

#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <ranges>
#include <sstream>
#include <string>
#include <type_traits>

namespace alpaka
{
    /** Simd vector
     *
     * @attention You should not use this type to create a buffer of SIMD vectors.
     * The implementation is not ABI compatible between different API's.
     * Using Simd data created on the host and used in the compute kernel will be undefined behaviour.
     *
     * This class is designed to be used via SimdPtr via reinterpretation of contiguous scalar data.
     *
     * @tparam T_Type data value type
     * @tparam T_width number of lanes in the SIMD vector
     * @tparam T_Storage wrapped native representation of the SIMD vector
     */
    template<
        typename T_Type,
        uint32_t T_width,
        typename T_Storage =
            /** do not use ALPAKA_TYPEOF(thisApi()) here else nvcc + gcc can trigger a compile error
             * error: use of built-in trait '__remove_cv(alpaka::api::Host)' in function signature;
             */
        typename alpaka::trait::GetSimdStorageType<decltype(thisApi()), T_Type, T_width>::type>
    struct Simd;

    namespace trait
    {
        template<typename T_Type, uint32_t T_width, typename T_Storage>
        struct IsSimd<Simd<T_Type, T_width, T_Storage>> : std::true_type
        {
        };
    } // namespace trait

    // friend forward declaration
    template<concepts::SimdMask T_Mask, concepts::Simd T_Simd>
    struct SimdWhereExpr;

    template<typename T_Type, uint32_t T_width, typename T_Storage>
    struct Simd : private T_Storage
    {
        using Storage = T_Storage;
        using value_type = T_Type;
        /** reference can be an implementation-detail proxy type. */
        using reference = typename T_Storage::reference;

        using index_type = uint32_t;
        using size_type = uint32_t;
        using rank_type = uint32_t;

        // universal vec used as fallback if T_Storage is holding the state in the template signature
        using UniSimd = Simd<T_Type, T_width>;

        /*Simds without elements are not allowed*/
        static_assert(T_width > 0u);

        constexpr Simd() = default;

        using Storage::asNativeType;

        /** static cast the instance to the storage type
         *
         * @attention: Do not use this method in user code, it is an implementation detail and can cause undefined
         * behaviour if used wrong.
         *
         * @{
         */
        constexpr auto& asStorage()
        {
            return static_cast<Storage&>(*this);
        }

        constexpr auto const& asStorage() const
        {
            return static_cast<Storage const&>(*this);
        }

        /** @} */

        /** Initialize via a generator expression
         *
         * The generator must return the value for the corresponding index of the component which is passed to the
         * generator.
         *
         * This constructor is not constexpr because std::simd is using a reinterpret_cast during the initialization
         * with a generator and complains that this is not allowed in constexpr functions.
         */
        template<typename F>
        requires std::invocable<F, std::integral_constant<uint32_t, 0u>>
        ALPAKA_FN_HOST_ACC explicit Simd(F&& generator)
            : Simd(std::forward<F>(generator), std::make_integer_sequence<uint32_t, T_width>{})
        {
            /* Do not change the enable if to `requires(std::is_invocable_v<F, std::integral_constant<uint32_t, 0u>>)`
             * nvcc 12.3.2 has a bug that creates compile issue when requires with std::is_invocable is used.
             */
        }

        /** Constructor for SIMD pack
         *
         * @attention This constructor allows implicit casts.
         *
         * This constructor is not constexpr because std::simd is using a reinterpret_cast during the initialization
         * with a generator and complains that this is not allowed in constexpr functions.
         *
         * @param args value of each lane index, x,y,z,...
         *
         */
        template<typename... T_Args>
        requires((std::is_convertible_v<T_Args, T_Type> && ...) && (sizeof...(T_Args) == T_width))
        ALPAKA_FN_HOST_ACC Simd(T_Args const&... args) : Storage(static_cast<T_Type>(args)...)
        {
        }

        constexpr Simd(Simd const& other) = default;

        constexpr Simd(T_Storage const& other) : T_Storage{other}
        {
        }

        /** constructor allows changing the storage policy
         */
        template<typename T_OtherStorage>
        constexpr Simd(Simd<T_Type, T_width, T_OtherStorage> const& other)
            : Simd([&](uint32_t const i) constexpr { return other[i]; })
        {
        }

        /** Allow static_cast / explicit cast to member type
         *
         * @attention only available for SIMD with a single lane.
         */
        constexpr explicit operator value_type() requires(T_width == 1u)
        {
            return (*this)[0];
        }

        /** Number of components/lanes in the SIMD pack. */
        static consteval uint32_t width()
        {
            return T_width;
        }

        constexpr void copyFrom(T_Type const* data, concepts::Alignment auto alignment)
        {
            Storage::copyFrom(data, alignment);
        }

        constexpr void copyTo(auto* data, concepts::Alignment auto alignment) const
        {
            Storage::copyTo(data, alignment);
        }

        /**
         * Creates a Simd where all lanes are set to the same value
         *
         * @param value Value which is set for all lanes
         * @return new Simd<...>
         */
        static constexpr auto fill(concepts::Convertible<T_Type> auto value)
        {
            /* Note the function is taking value as copy because it is typically a scalar value.
             * If a const reference is used, it would not be possible to pass a host defined constexpr value into
             * fill() when CUDA is used. Issue was seen with nvcc CUDA 13.0.2. see
             * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#constexpr-variables
             */
            return Simd{Storage::fill(static_cast<T_Type>(value))};
        }

        constexpr Simd toRT() const
        {
            return *this;
        }

        constexpr Simd revert() const
        {
            Simd invertedSimd{};
            for(uint32_t i = 0u; i < T_width; i++)
                invertedSimd[T_width - 1 - i] = (*this)[i];

            return invertedSimd;
        }

        constexpr Simd& operator=(Simd const&) = default;
        constexpr Simd& operator=(Simd&&) = default;

        constexpr Simd operator-() const
        {
            return Simd([this](uint32_t const i) constexpr { return -(*this)[i]; });
        }

        /** assign operator
         * @{
         */
#define ALPAKA_VECTOR_ASSIGN_OP(op)                                                                                   \
    template<typename T_OtherStorage>                                                                                 \
    constexpr Simd& operator op(Simd<T_Type, T_width, T_OtherStorage> const& rhs)                                     \
    {                                                                                                                 \
        this->asStorage() op rhs.asStorage();                                                                         \
        return *this;                                                                                                 \
    }                                                                                                                 \
    constexpr Simd& operator op(concepts::LosslesslyConvertible<T_Type> auto const value)                             \
    {                                                                                                                 \
        this->asStorage() op static_cast<T_Type>(value);                                                              \
        return *this;                                                                                                 \
    }

        ALPAKA_VECTOR_ASSIGN_OP(+=)
        ALPAKA_VECTOR_ASSIGN_OP(-=)
        ALPAKA_VECTOR_ASSIGN_OP(/=)
        ALPAKA_VECTOR_ASSIGN_OP(*=)
        ALPAKA_VECTOR_ASSIGN_OP(=)

#undef ALPAKA_VECTOR_ASSIGN_OP

        /** @} */

        /** access a lane by index
         *
         * @return The returned type is implementation specific, therefore it can be a proxy reference.
         *         You can not use the returned value to deduct the type and assume that it will be the value type of
         * Simd.
         */
        constexpr reference operator[](std::integral auto const idx)
        {
            return asStorage()[idx];
        }

        /** access a lane by index
         *
         * @return The value type, by copy.
         */
        constexpr value_type operator[](std::integral auto const idx) const
        {
            return asStorage()[idx];
        }

#define ALPAKA_NAMED_ARRAY_ACCESS(functionName, laneIdx)                                                              \
    constexpr reference functionName() requires(T_width >= laneIdx + 1)                                               \
    {                                                                                                                 \
        return (*this)[laneIdx];                                                                                      \
    }                                                                                                                 \
    constexpr value_type functionName() const requires(T_width >= laneIdx + 1)                                        \
    {                                                                                                                 \
        return (*this)[laneIdx];                                                                                      \
    }

        /** @brief named lane access
         *
         * @attention The mapping from names x,y,z,w to memory indices differ from the mapping of an alpaka vector @c
         * Vec. The availability of the naming methods depends on the SIMD width.
         *
         * You can have access to the same lane index via different nonspecific naming.
         *
         * @code
         * lane index   :  0,  1,  2,  3, ...,  9, 10, ... , 15
         * hexadecimal  : s0, s1, s2, s3, ..., s9, SA, ... , SF
         * coordinate   :  x,  y,  z,  w
         * color channel:  r,  g,  b,  a
         * @endcode
         *
         * @{
         */
        ALPAKA_NAMED_ARRAY_ACCESS(x, 0u)
        ALPAKA_NAMED_ARRAY_ACCESS(y, 1u)
        ALPAKA_NAMED_ARRAY_ACCESS(z, 2u)
        ALPAKA_NAMED_ARRAY_ACCESS(w, 3u)

        ALPAKA_NAMED_ARRAY_ACCESS(r, 0u)
        ALPAKA_NAMED_ARRAY_ACCESS(g, 1u)
        ALPAKA_NAMED_ARRAY_ACCESS(b, 2u)
        ALPAKA_NAMED_ARRAY_ACCESS(a, 3u)

        ALPAKA_NAMED_ARRAY_ACCESS(s0, 0u)
        ALPAKA_NAMED_ARRAY_ACCESS(s1, 1u)
        ALPAKA_NAMED_ARRAY_ACCESS(s2, 2u)
        ALPAKA_NAMED_ARRAY_ACCESS(s3, 3u)
        ALPAKA_NAMED_ARRAY_ACCESS(s4, 4u)
        ALPAKA_NAMED_ARRAY_ACCESS(s5, 5u)
        ALPAKA_NAMED_ARRAY_ACCESS(s6, 6u)
        ALPAKA_NAMED_ARRAY_ACCESS(s7, 7u)
        ALPAKA_NAMED_ARRAY_ACCESS(s8, 8u)
        ALPAKA_NAMED_ARRAY_ACCESS(s9, 9u)
        ALPAKA_NAMED_ARRAY_ACCESS(sA, 10u)
        ALPAKA_NAMED_ARRAY_ACCESS(sB, 11u)
        ALPAKA_NAMED_ARRAY_ACCESS(sC, 12u)
        ALPAKA_NAMED_ARRAY_ACCESS(sD, 13u)
        ALPAKA_NAMED_ARRAY_ACCESS(sE, 14u)
        ALPAKA_NAMED_ARRAY_ACCESS(sF, 15u)
        /** @} */

#undef ALPAKA_NAMED_ARRAY_ACCESS

        /** Shrink the number of elements of a vector.
         *
         * Highest indices kept alive.
         *
         * @tparam T_numElements New width of the SIMD pack.
         * @return First T_numElements elements of the origin vector
         */
        template<uint32_t T_numElements>
        constexpr Simd<T_Type, T_numElements> rshrink() const
        {
            static_assert(T_numElements <= T_width);
            Simd<T_Type, T_numElements> result{};
            for(uint32_t i = 0u; i < T_numElements; i++)
                result[T_numElements - 1u - i] = (*this)[T_width - 1u - i];

            return result;
        }

        /** Shrink the SIMD pack
         *
         * Removes the last value.
         */
        constexpr Simd<T_Type, T_width - 1u> eraseBack() const requires(T_width > 1u)
        {
            constexpr auto reducedDim = T_width - 1u;
            Simd<T_Type, reducedDim> result{};
            for(uint32_t i = 0u; i < reducedDim; i++)
                result[i] = (*this)[i];

            return result;
        }

        /** Shrink the number of elements of a vector.
         *
         * @tparam T_numElements New width of the SIMD pack.
         * @param startIdx Index within the origin vector which will be the last element in the result.
         * @return T_numElements elements of the origin vector starting with the index startIdx.
         *         Indexing will wrapp around when the begin of the origin vector is reached.
         */
        template<uint32_t T_numElements>
        constexpr Simd<value_type, T_numElements> rshrink(std::integral auto const startIdx) const
        {
            static_assert(T_numElements <= T_width);
            Simd<value_type, T_numElements> result;
            for(uint32_t i = 0u; i < T_numElements; i++)
                result[T_numElements - 1u - i] = (*this)[(T_width + startIdx - i) % T_width];
            return result;
        }

        /** Removes a component
         *
         * It is not allowed to call this method on a vector with the width of one.
         *
         * @tparam T_laneIdxToRemove index which shall be removed; range: [ 0; T_width - 1 ]
         * @return vector with `T_width - 1` elements
         */
        template<std::integral auto T_laneIdxToRemove>
        constexpr Simd<value_type, T_width - 1u> remove() const requires(T_width >= 2u)
        {
            Simd<value_type, T_width - 1u> result{};
            for(int i = 0u; i < static_cast<int>(T_width - 1u); ++i)
            {
                // skip component which must be deleted
                int const sourceIdx = i >= static_cast<int>(T_laneIdxToRemove) ? i + 1 : i;
                result[i] = (*this)[sourceIdx];
            }
            return result;
        }

        /** Returns product of all components.
         *
         * @return product of components
         */
        [[nodiscard]] constexpr value_type product() const
        {
            return reduce(std::multiplies{});
        }

        /** Returns sum of all components.
         *
         * @return sum of components
         */
        [[nodiscard]] constexpr value_type sum() const
        {
            return reduce(std::plus{});
        }

        /** reduce all elements to a single value
         *
         * For better numerical stability a tree reduce algorithm is used.
         *
         * @tparam BinaryOp binary functor executed to reduce the range
         *                  The binary operation must be associative.
         * @return the type of the result depends on the binary functor
         */
        [[nodiscard]] constexpr auto reduce(auto&& reduceFunc) const
            -> decltype(reduceFunc(std::declval<value_type>(), std::declval<value_type>()))
        {
            return reduceRange(ALPAKA_FORWARD(reduceFunc));
        }

        template<typename T_OtherStorage>
        constexpr auto min(Simd<T_Type, T_width, T_OtherStorage> const& rhs) const
        {
            Simd result{};
            for(uint32_t d = 0u; d < T_width; d++)
                result[d] = std::min((*this)[d], rhs[d]);
            return result;
        }

        /** create string out of the SIMD pack
         *
         * @param separator string to separate components of the SIMD pack
         * @param enclosings string with width 2 to enclose SIMD pack
         *                   width == 0 ? no enclose symbols
         *                   width == 1 ? means enclose symbol begin and end are equal
         *                   width >= 2 ? letter[0] = begin enclose symbol
         *                               letter[1] = end enclose symbol
         *
         * example:
         * .toString(";","|")     -> |x;...;z|
         * .toString(",","[]")    -> [x,...,z]
         */
        std::string toString(std::string const separator = ",", std::string const enclosings = "{}") const
        {
            std::string localeEnclosingBegin;
            std::string localeEnclosingEnd;
            size_t enclosingLaneIdx = enclosings.size();

            if(enclosingLaneIdx > 0)
            {
                /* % avoid out of memory access */
                localeEnclosingBegin = enclosings[0 % enclosingLaneIdx];
                localeEnclosingEnd = enclosings[1 % enclosingLaneIdx];
            }

            std::stringstream stream;
            stream << localeEnclosingBegin << (*this)[0];

            for(uint32_t i = 1u; i < T_width; ++i)
                stream << separator << (*this)[i];
            stream << localeEnclosingEnd;
            return stream.str();
        }

    private:
        template<typename F, uint32_t... Is>
        constexpr explicit Simd(F&& generator, std::integer_sequence<uint32_t, Is...>)
            : Storage{generator(std::integral_constant<uint32_t, Is>{})...}
        {
        }

        /** reduce over a range of elements
         *
         * @tparam BinaryOp binary functor executed to reduce the range
         * @tparam T_start start index
         * @tparam T_end end index (excluded)
         * @return the type of the result depends on the binary functor
         */
        template<uint32_t T_start = 0u, uint32_t T_end = width()>
        [[nodiscard]] constexpr auto reduceRange(auto&& reduceFunc) const
            -> decltype(reduceFunc(std::declval<value_type>(), std::declval<value_type>()))
        {
            // elements in the range
            constexpr uint32_t size = T_end - T_start;
            // single element termination
            if constexpr(size == 1u)
            {
                return (*this)[T_start];
            }
#if ALPAKA_LANG_SYCL
            // SYCL can not call recursive functions
            auto result = (*this)[T_start];
            for(uint32_t i = T_start + 1u; i < T_end; ++i)
            {
                result = reduceFunc(result, (*this)[i]);
            }
            return result;
#else
            // split range at midpoint
            constexpr uint32_t mid = T_start + size / 2u;

            // recursively reduce both halves and combine
            return reduceFunc(
                reduceRange<T_start, mid>(ALPAKA_FORWARD(reduceFunc)),
                reduceRange<mid, T_end>(ALPAKA_FORWARD(reduceFunc)));
#endif
        }

        template<concepts::SimdMask T_Mask, concepts::Simd T_Simd>
        friend struct SimdWhereExpr;
    };

    template<std::size_t I, typename T_Type, uint32_t T_width, typename T_Storage>
    constexpr auto get(Simd<T_Type, T_width, T_Storage> const& v)
    {
        return v[I];
    }

    template<std::size_t I, typename T_Type, uint32_t T_width, typename T_Storage>
    constexpr auto& get(Simd<T_Type, T_width, T_Storage>& v)
    {
        return v[I];
    }

    template<typename T_Type, uint32_t T_width, typename T_Storage>
    std::ostream& operator<<(std::ostream& s, Simd<T_Type, T_width, T_Storage> const& vec)
    {
        return s << vec.toString();
    }

    // type deduction guide
    template<typename T_1, typename... T_Args>
    ALPAKA_FN_DG Simd(T_1, T_Args...) -> Simd<T_1, uint32_t(sizeof...(T_Args) + 1u)>;

    /** binary operators
     * @{
     */
#define ALPAKA_VECTOR_BINARY_OP(typenameOrConcept, op)                                                                \
    template<typenameOrConcept T_Type, uint32_t T_width, typename T_Storage, typename T_OtherStorage>                 \
    constexpr auto operator op(                                                                                       \
        const Simd<T_Type, T_width, T_Storage>& lhs,                                                                  \
        const Simd<T_Type, T_width, T_OtherStorage>& rhs)                                                             \
    {                                                                                                                 \
        using StoreageType = ALPAKA_TYPEOF(lhs.asStorage() op rhs.asStorage());                                       \
        return Simd<T_Type, T_width, StoreageType>(lhs.asStorage() op rhs.asStorage());                               \
    }                                                                                                                 \
    template<                                                                                                         \
        typenameOrConcept T_Type,                                                                                     \
        concepts::LosslesslyConvertible<T_Type> T_ValueType,                                                          \
        uint32_t T_width,                                                                                             \
        typename T_Storage>                                                                                           \
    constexpr auto operator op(const Simd<T_Type, T_width, T_Storage>& lhs, T_ValueType rhs)                          \
    {                                                                                                                 \
        using StoreageType = ALPAKA_TYPEOF(lhs.asStorage() op static_cast<T_Type>(rhs));                              \
        return Simd<T_Type, T_width, StoreageType>(lhs.asStorage() op static_cast<T_Type>(rhs));                      \
    }                                                                                                                 \
    template<                                                                                                         \
        typenameOrConcept T_Type,                                                                                     \
        concepts::LosslesslyConvertible<T_Type> T_ValueType,                                                          \
        uint32_t T_width,                                                                                             \
        typename T_Storage>                                                                                           \
    constexpr auto operator op(T_ValueType lhs, const Simd<T_Type, T_width, T_Storage>& rhs)                          \
    {                                                                                                                 \
        using StoreageType = ALPAKA_TYPEOF(static_cast<T_Type>(lhs) op rhs.asStorage());                              \
        return Simd<T_Type, T_width, StoreageType>(static_cast<T_Type>(lhs) op rhs.asStorage());                      \
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


#define ALPAKA_VECTOR_BINARY_CMP_OP(typenameOrConcept, op)                                                            \
    template<typenameOrConcept T_Type, uint32_t T_width, typename T_Storage, typename T_OtherStorage>                 \
    constexpr auto operator op(                                                                                       \
        const Simd<T_Type, T_width, T_Storage>& lhs,                                                                  \
        const Simd<T_Type, T_width, T_OtherStorage>& rhs)                                                             \
    {                                                                                                                 \
        using StoreageType = ALPAKA_TYPEOF(lhs.asStorage() op rhs.asStorage());                                       \
        return SimdMask<T_Type, T_width, StoreageType>(lhs.asStorage() op rhs.asStorage());                           \
    }                                                                                                                 \
    template<                                                                                                         \
        typenameOrConcept T_Type,                                                                                     \
        concepts::LosslesslyConvertible<T_Type> T_ValueType,                                                          \
        uint32_t T_width,                                                                                             \
        typename T_Storage>                                                                                           \
    constexpr auto operator op(const Simd<T_Type, T_width, T_Storage>& lhs, T_ValueType rhs)                          \
    {                                                                                                                 \
        using StoreageType = ALPAKA_TYPEOF(lhs.asStorage() op static_cast<T_Type>(rhs));                              \
        return SimdMask<T_Type, T_width, StoreageType>(lhs.asStorage() op static_cast<T_Type>(rhs));                  \
    }                                                                                                                 \
    template<                                                                                                         \
        typenameOrConcept T_Type,                                                                                     \
        concepts::LosslesslyConvertible<T_Type> T_ValueType,                                                          \
        uint32_t T_width,                                                                                             \
        typename T_Storage>                                                                                           \
    constexpr auto operator op(T_ValueType lhs, const Simd<T_Type, T_width, T_Storage>& rhs)                          \
    {                                                                                                                 \
        using StoreageType = ALPAKA_TYPEOF(static_cast<T_Type>(lhs) op rhs.asStorage());                              \
        return SimdMask<T_Type, T_width, StoreageType>(static_cast<T_Type>(lhs) op rhs.asStorage());                  \
    }

    ALPAKA_VECTOR_BINARY_CMP_OP(typename, >=)
    ALPAKA_VECTOR_BINARY_CMP_OP(typename, >)
    ALPAKA_VECTOR_BINARY_CMP_OP(typename, <=)
    ALPAKA_VECTOR_BINARY_CMP_OP(typename, <)
    ALPAKA_VECTOR_BINARY_CMP_OP(typename, ==)
    ALPAKA_VECTOR_BINARY_CMP_OP(typename, !=)

#undef ALPAKA_VECTOR_BINARY_CMP_OP

    /** @} */


    namespace trait
    {
        template<typename T_Type, uint32_t T_width, typename T_Storage>
        struct GetDim<alpaka::Simd<T_Type, T_width, T_Storage>>
        {
            static constexpr uint32_t value = T_width;
        };

        template<typename T_Type, uint32_t T_width, typename T_Storage>
        struct GetValueType<alpaka::Simd<T_Type, T_width, T_Storage>>
        {
            using type = T_Type;
        };
    } // namespace trait

    namespace internal
    {
        template<typename T_To, typename T_Type, uint32_t T_width, typename T_Storage>
        struct PCast::Op<T_To, alpaka::Simd<T_Type, T_width, T_Storage>>
        {
            constexpr auto operator()(auto&& input) const
                requires std::convertible_to<T_Type, T_To> && (!std::same_as<T_To, T_Type>)
            {
                return typename alpaka::Simd<T_To, T_width, T_Storage>::UniSimd(
                    [&](uint32_t idx) constexpr { return static_cast<T_To>(input[idx]); });
            }

            constexpr decltype(auto) operator()(auto&& input) const requires std::same_as<T_To, T_Type>
            {
                return std::forward<decltype(input)>(input);
            }
        };
    } // namespace internal
}; // namespace alpaka

namespace std
{
    template<typename T_Type, uint32_t T_width, typename T_Storage>
    struct tuple_size<alpaka::Simd<T_Type, T_width, T_Storage>>
    {
        static constexpr std::size_t value = T_width;
    };

    template<std::size_t I, typename T_Type, uint32_t T_width, typename T_Storage>
    struct tuple_element<I, alpaka::Simd<T_Type, T_width, T_Storage>>
    {
        using type = T_Type;
    };
} // namespace std
