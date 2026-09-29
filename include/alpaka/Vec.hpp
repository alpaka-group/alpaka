/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/cast.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/core/util.hpp"
#include "alpaka/trait.hpp"
#include "unused.hpp"

#include <array>
#include <concepts>
#include <cstdint>
#include <iosfwd>
#include <ranges>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

namespace alpaka
{
    namespace trait
    {
        template<typename T>
        struct IsVector : std::false_type
        {
        };

        template<typename T>
        struct IsCVector : std::false_type
        {
        };
    } // namespace trait

    template<typename T>
    constexpr bool isVector_v = trait::IsVector<T>::value;

    template<typename T>
    constexpr bool isCVector_v = trait::IsCVector<T>::value;

    namespace concepts
    {

        /** Concept to check if a type is a vector
         *
         * @tparam T Type to check
         * @tparam T_ValueType enforce a value type of the vector, if not provided the value type is not checked
         * @tparam T_dim enforce a dimensionality of the vector, if not provided the value is not checked
         */
        template<typename T, typename T_ValueType = alpaka::NotRequired, uint32_t T_dim = alpaka::notRequiredDim>
        concept Vector = isVector_v<T>
                         && (std::same_as<T_ValueType, GetValueType_t<std::decay_t<T>>>
                             || std::same_as<T_ValueType, alpaka::NotRequired>)
                         && ((T_dim == alpaka::notRequiredDim) || (T::dim() == T_dim));

        /** Concept to check if a type is a vector or scalar variable
         *
         * @tparam T Type to check
         * @tparam T_ValueType enforce a value type of T, if not provided the value type is not checked
         */
        template<typename T, typename T_ValueType = alpaka::NotRequired>
        concept VectorOrScalar = (isVector_v<T> || std::integral<T> || std::floating_point<T>)
                                 && (std::same_as<T_ValueType, GetValueType_t<std::decay_t<T>>>
                                     || std::same_as<T_ValueType, alpaka::NotRequired>);

        /** Concept to check if a type is a CVector
         *
         * @details
         * Checks whether the given type is a CVector. For more information, refer to the implementation alpaka::CVec.
         */
        template<typename T, typename T_ValueType = alpaka::NotRequired>
        concept CVector = isCVector_v<T>
                          && (std::same_as<T_ValueType, GetValueType_t<std::decay_t<T>>>
                              || std::same_as<T_ValueType, alpaka::NotRequired>);

        /** Concept to check if a type is a vector or a specific other type
         *
         * @tparam T Type to check
         * @tparam T_RequiredComponent enforce that T is a vector or a specific other type
         */
        template<typename T, typename T_RequiredComponent>
        concept TypeOrVector = (isVector_v<T> || std::is_same_v<T, T_RequiredComponent>);

        template<typename T, typename T_RequiredComponent>
        concept VectorOrConvertibleType = (isVector_v<T> || std::is_convertible_v<T, T_RequiredComponent>);
    } // namespace concepts

    /** Array storge for vector data
     *
     * This class is a workaround and is simply wrapping std::array. It is required because the dim in std::array
     * in the template signature is size_t. This produces template deduction issues for math::Vec if we sue
     * array as default storage without this wrapper.
     */
    template<typename T_Type, uint32_t T_dim>
    struct ArrayStorage : protected std::array<T_Type, T_dim>
    {
        using value_type = T_Type;
        using BaseType = std::array<T_Type, T_dim>;
        using BaseType::operator[];

        // constructor is required because exposing the array constructors does not work
        template<typename... T_Args>
        constexpr ArrayStorage(T_Args&&... args) : BaseType{std::forward<T_Args>(args)...}
        {
        }

        constexpr ArrayStorage(std::array<T_Type, T_dim> const& data) : BaseType{data}
        {
        }
    };

    namespace detail
    {
        template<typename T, T... T_values>
        struct CVec
        {
            using value_type = T;

            static consteval uint32_t dim()
            {
                return sizeof...(T_values);
            }

            constexpr T operator[](std::integral auto const idx) const
            {
                // default initializes with first value
                T result = std::get<0>(std::forward_as_tuple(T_values...));

                if constexpr(dim() > 1u)
                {
                    [[maybe_unused]] bool _ = std::apply(
                        [idx, &result](auto&&, auto&&... values) constexpr
                        {
                            using IdxType = ALPAKA_TYPEOF(idx);
                            IdxType i{1u};
                            return ((idx == i++ && (result = values, true)) || ...);
                        },
                        std::forward_as_tuple(T_values...));
                }
                return result;
            }

            template<T T_value>
            static constexpr auto fill()
            {
                using IotaSeq = std::make_integer_sequence<T, dim()>;
                return integerSequenceToCVec(IotaSeq{}, [](auto&&) constexpr { return T_value; });
            }

        private:
            template<T... T_indices>
            static constexpr auto integerSequenceToCVec(
                std::integer_sequence<T, T_indices...>,
                auto const op = std::identity{})
            {
                return CVec<T, op(T_indices)...>{};
            }
        };

        template<typename T>
        struct TemplateSignatureStorage : std::false_type
        {
        };

        template<typename T_Type, T_Type... T_values>
        struct TemplateSignatureStorage<CVec<T_Type, T_values...>> : std::true_type
        {
        };

        template<typename T>
        constexpr bool templateSignatureStorage_v = TemplateSignatureStorage<T>::value;
    } // namespace detail

    template<typename T_Type, uint32_t T_dim, typename T_Storage = alpaka::ArrayStorage<T_Type, T_dim>>
    struct Vec : private T_Storage
    {
        using Storage = T_Storage;
        using value_type = T_Type;
        using ParamType = value_type;

        using index_type = uint32_t;
        using size_type = uint32_t;
        using rank_type = uint32_t;

        // universal vec used as fallback if T_Storage is holding the state in the template signature
        using UniVec = Vec<T_Type, T_dim>;

        /*Vecs without elements are not allowed*/
        static_assert(T_dim > 0u);

        constexpr Vec() = default;

        /** Initialize via a generator expression
         *
         * The generator must return the value for the corresponding index of the component which is passed to the
         * generator.
         */
        template<
            typename F,
            std::enable_if_t<std::is_invocable_v<F, std::integral_constant<uint32_t, 0u>>, uint32_t> = 0u>
        constexpr explicit Vec(F&& generator)
            : Vec(std::forward<F>(generator), std::make_integer_sequence<uint32_t, T_dim>{})
        {
        }

    private:
        template<typename F, uint32_t... Is>
        constexpr explicit Vec(F&& generator, std::integer_sequence<uint32_t, Is...>)
            : Storage{generator(std::integral_constant<uint32_t, Is>{})...}
        {
        }

    public:
        /** Constructor for N-dimensional vector
         *
         * @attention This constructor allows implicit casts.
         *
         * @param args value of each dimension, x,y,z,...
         *
         * A constexpr vector should be initialized with {} instead of () because at least
         * CUDA 11.6 has problems in cases where a compile time evaluation is required.
         * @code{.cpp}
         *   constexpr auto vec1 = Vec{ 1 };
         *   constexpr auto vec2 = Vec{ 1, 2 };
         *   //or explicit
         *   constexpr auto vec3 = Vec<int, 3u>{ 1, 2, 3 };
         *   constexpr auto vec4 = Vec<int, 3u>{ {1, 2, 3} };
         * @endcode
         */
        template<typename... T_Args>
        requires(std::is_convertible_v<T_Args, T_Type> && ...)
        constexpr Vec(T_Args const&... args) : Storage(static_cast<T_Type>(args)...)
        {
        }

        constexpr Vec(Vec const& other) = default;

        constexpr Vec(T_Storage const& other) : T_Storage{other}
        {
        }

        /** constructor allows changing the storage policy
         */
        template<typename T_OtherStorage>
        constexpr Vec(Vec<T_Type, T_dim, T_OtherStorage> const& other)
            : Vec([&](uint32_t const i) constexpr { return other[i]; })
        {
        }

        /** Allow static_cast / explicit cast to member type for 1D vector */
        template<uint32_t T_deferDim = T_dim, typename = typename std::enable_if<T_deferDim == 1u>::type>
        constexpr explicit operator value_type()
        {
            return (*this)[0];
        }

        static consteval uint32_t dim()
        {
            return T_dim;
        }

        /**
         * Creates a Vec where all dimensions are set to the same value
         *
         * @param value Value which is set for all dimensions
         * @return new Vec<...>
         */
        static constexpr auto fill(concepts::Convertible<T_Type> auto const& value)
        {
            if constexpr(requires { detail::templateSignatureStorage_v<T_Storage>; })
            {
                return UniVec([=](uint32_t const) { return static_cast<T_Type>(value); });
            }
            else
            {
                return Vec([=](uint32_t const) { return static_cast<T_Type>(value); });
            }
        }

        template<auto T_v>
        requires(isConvertible_v<ALPAKA_TYPEOF(T_v), T_Type>)
        static constexpr auto fill() requires requires { T_Storage::template fill<T_v>(); }
        {
            return Vec<T_Type, T_dim, ALPAKA_TYPEOF(T_Storage::template fill<static_cast<T_Type>(T_v)>())>{};
        }

        constexpr Vec toRT() const

        {
            return *this;
        }

        constexpr Vec revert() const
        {
            Vec invertedVec{};
            for(uint32_t i = 0u; i < T_dim; i++)
                invertedVec[T_dim - 1 - i] = (*this)[i];

            return invertedVec;
        }

        constexpr Vec& operator=(Vec const&) = default;
        constexpr Vec& operator=(Vec&&) = default;

        constexpr Vec operator-() const
        {
            return Vec([this](uint32_t const i) constexpr { return -(*this)[i]; });
        }

#define ALPAKA_VECTOR_ASSIGN_OP(op)                                                                                   \
    template<typename T_OtherStorage>                                                                                 \
    constexpr Vec& operator op(Vec<T_Type, T_dim, T_OtherStorage> const& rhs)                                         \
    {                                                                                                                 \
        for(uint32_t i = 0u; i < T_dim; i++)                                                                          \
        {                                                                                                             \
            if constexpr(requires { unWrapp((*this)[i]) op rhs[i]; })                                                 \
            {                                                                                                         \
                unWrapp((*this)[i]) op rhs[i];                                                                        \
            }                                                                                                         \
            else                                                                                                      \
            {                                                                                                         \
                (*this)[i] op rhs[i];                                                                                 \
            }                                                                                                         \
        }                                                                                                             \
        return *this;                                                                                                 \
    }                                                                                                                 \
    constexpr Vec& operator op(concepts::LosslesslyConvertible<T_Type> auto const value)                              \
    {                                                                                                                 \
        for(uint32_t i = 0u; i < T_dim; i++)                                                                          \
        {                                                                                                             \
            if constexpr(requires { unWrapp((*this)[i]) op value; })                                                  \
            {                                                                                                         \
                unWrapp((*this)[i]) op value;                                                                         \
            }                                                                                                         \
            else                                                                                                      \
            {                                                                                                         \
                (*this)[i] op value;                                                                                  \
            }                                                                                                         \
        }                                                                                                             \
        return *this;                                                                                                 \
    }

        /** assign operator
         * @{
         */
        ALPAKA_VECTOR_ASSIGN_OP(+=)
        ALPAKA_VECTOR_ASSIGN_OP(-=)
        ALPAKA_VECTOR_ASSIGN_OP(/=)
        ALPAKA_VECTOR_ASSIGN_OP(*=)
        ALPAKA_VECTOR_ASSIGN_OP(=)
        /** @} */

#undef ALPAKA_VECTOR_ASSIGN_OP

        constexpr decltype(auto) operator[](std::integral auto const idx)
        {
            return Storage::operator[](idx);
        }

        constexpr decltype(auto) operator[](std::integral auto const idx) const
        {
            return Storage::operator[](idx);
        }

#define ALPAKA_NAMED_ARRAY_ACCESS(functionName, indexPos)                                                             \
    /* An integer underflow may occur with `indexPos`, for example if `T_dim` is equal to 1 and `y()` should be       \
     declared as `(T_dim - 2u)`. Therefore the `requires` is fine, because everything which should be theoretical     \
     negative become much bigger.                                                                                     \
     * than T_dim. */                                                                                                 \
    constexpr decltype(auto) functionName() requires((indexPos) < T_dim)                                              \
    {                                                                                                                 \
        return (*this)[indexPos];                                                                                     \
    }                                                                                                                 \
    constexpr decltype(auto) functionName() const requires((indexPos) < T_dim)                                        \
    {                                                                                                                 \
        return (*this)[indexPos];                                                                                     \
    }

        /** named member access
         *
         * index -> name [0->w, 1->z, 2->y, 3->x]
         * @{
         */
        ALPAKA_NAMED_ARRAY_ACCESS(x, T_dim - 1u)
        ALPAKA_NAMED_ARRAY_ACCESS(y, T_dim - 2u)
        ALPAKA_NAMED_ARRAY_ACCESS(z, T_dim - 3u)
        ALPAKA_NAMED_ARRAY_ACCESS(w, T_dim - 4u)
        /** @} */

#undef ALPAKA_NAMED_ARRAY_ACCESS

        constexpr decltype(auto) back()
        {
            return (*this)[T_dim - 1u];
        }

        constexpr decltype(auto) back() const
        {
            return (*this)[T_dim - 1u];
        }

        /** Shrink the number of elements of a vector.
         *
         * Highest indices kept alive.
         *
         * @tparam T_numElements New dimension of the vector.
         * @return First T_numElements elements of the origin vector
         */
        template<uint32_t T_numElements>
        constexpr Vec<T_Type, T_numElements> rshrink() const
        {
            static_assert(T_numElements <= T_dim);
            Vec<T_Type, T_numElements> result{};
            for(uint32_t i = 0u; i < T_numElements; i++)
                result[T_numElements - 1u - i] = (*this)[T_dim - 1u - i];

            return result;
        }

        /** Shrink the vector
         *
         * Removes the last value.
         */
        constexpr Vec<T_Type, T_dim - 1u> eraseBack() const requires(T_dim > 1u)
        {
            constexpr auto reducedDim = T_dim - 1u;
            Vec<T_Type, reducedDim> result{};
            for(uint32_t i = 0u; i < reducedDim; i++)
                result[i] = (*this)[i];

            return result;
        }

        /** Shrink the number of elements of a vector.
         *
         * @tparam T_numElements New dimension of the vector.
         * @param startIdx Index within the origin vector which will be the last element in the result.
         * @return T_numElements elements of the origin vector starting with the index startIdx.
         *         Indexing will wrapp around when the begin of the origin vector is reached.
         */
        template<uint32_t T_numElements>
        constexpr Vec<value_type, T_numElements> rshrink(std::integral auto const startIdx) const
        {
            static_assert(T_numElements <= T_dim);
            Vec<value_type, T_numElements> result;
            for(uint32_t i = 0u; i < T_numElements; i++)
                result[T_numElements - 1u - i] = (*this)[(T_dim + startIdx - i) % T_dim];
            return result;
        }

        /** Assign an value to the given index position
         *
         * @tparam T_elementIdx Index of the element from the begin which shall be replaced; range: [ 0; T_dim - 1 ]
         * @param value value to assign to the element at the given index position
         * @return copy of the vector with where the index positions are updated with value
         */
        template<uint32_t T_elementIdx = 0>
        constexpr Vec<T_Type, T_dim> assign(T_Type const& value) const requires(T_elementIdx < T_dim)
        {
            auto result = *this;
            result[T_elementIdx] = value;
            return result;
        }

        /** Assign an value to the given index position
         *
         * @param selection CVec with the indices of the elements which shall be replaced; indices range must be
         * [0; T_dim -1]
         * @param value value to assign to the element at the given index position
         * @return copy of the vector with where the index positions are updated with value
         */
        constexpr Vec<T_Type, T_dim> assign(
            concepts::CVector auto const selection,
            concepts::Vector<T_Type> auto const& value) const requires(ALPAKA_TYPEOF(value)::dim() <= T_dim)
        {
            auto result = *this;
            result.ref(selection) = value;
            return result;
        }

        /** Assign an value to the given index position
         *
         * @tparam T_elementIdx Index of the element from the back which shall be replaced; range: [ 0; T_dim - 1 ]
         * @param value value to assign to the element at the given index position
         * @return copy of the vector with where the index positions are updated with value
         */
        template<uint32_t T_elementIdx = T_dim - 1u>
        constexpr Vec<T_Type, T_dim> rAssign(T_Type const& value) const requires(T_elementIdx < T_dim)
        {
            Vec<T_Type, T_dim> result = *this;
            result[T_elementIdx] = value;
            return result;
        }

        /** Removes a component
         *
         * It is not allowed to call this method on a vector with the dimensionality of one.
         *
         * @tparam T_dimToRemove index which shall be removed; range: [ 0; T_dim - 1 ]
         * @return vector with `T_dim - 1` elements
         */
        template<std::integral auto T_dimToRemove>
        constexpr Vec<value_type, T_dim - 1u> remove() const requires(T_dim >= 2u)
        {
            Vec<value_type, T_dim - 1u> result{};
            for(int i = 0u; i < static_cast<int>(T_dim - 1u); ++i)
            {
                // skip component which must be deleted
                int const sourceIdx = i >= static_cast<int>(T_dimToRemove) ? i + 1 : i;
                result[i] = (*this)[sourceIdx];
            }
            return result;
        }

        /** Returns product of all components.
         *
         * @return product of components
         */
        constexpr value_type product() const
        {
            value_type result = (*this)[0];
            for(uint32_t i = 1u; i < T_dim; i++)
                result *= (*this)[i];
            return result;
        }

        /** Returns sum of all components.
         *
         * @return sum of components
         */
        constexpr value_type sum() const
        {
            value_type result = (*this)[0];
            for(uint32_t i = 1u; i < T_dim; i++)
                result += (*this)[i];
            return result;
        }

        /**
         * == comparison operator.
         *
         * Compares dims of two DataSpaces.
         *
         * @param other Vec to compare to
         * @return true if all components in both vectors are equal, else false
         */
        template<typename T_OtherStorage>
        constexpr bool operator==(Vec<T_Type, T_dim, T_OtherStorage> const& rhs) const
        {
            bool result = true;
            for(uint32_t i = 0u; i < T_dim; i++)
                result = result && ((*this)[i] == rhs[i]);
            return result;
        }

        /**
         * != comparison operator.
         *
         * Compares dims of two DataSpaces.
         *
         * @param other Vec to compare to
         * @return true if one component in both vectors are not equal, else false
         */
        template<typename T_OtherStorage>
        constexpr bool operator!=(Vec<T_Type, T_dim, T_OtherStorage> const& rhs) const
        {
            return !((*this) == rhs);
        }

        template<typename T_OtherStorage>
        constexpr auto min(Vec<T_Type, T_dim, T_OtherStorage> const& rhs) const
        {
            typename Vec::UniVec result{};
            for(uint32_t d = 0u; d < T_dim; d++)
                result[d] = std::min((*this)[d], rhs[d]);
            return result;
        }

        /** create string out of the vector
         *
         * @param separator string to separate components of the vector
         * @param enclosings string with dim 2 to enclose vector
         *                   dim == 0 ? no enclose symbols
         *                   dim == 1 ? means enclose symbol begin and end are equal
         *                   dim >= 2 ? letter[0] = begin enclose symbol
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
            size_t enclosingDim = enclosings.size();

            if(enclosingDim > 0)
            {
                /* % avoid out of memory access */
                localeEnclosingBegin = enclosings[0 % enclosingDim];
                localeEnclosingEnd = enclosings[1 % enclosingDim];
            }

            std::stringstream stream;
            stream << localeEnclosingBegin << (*this)[0];

            for(uint32_t i = 1u; i < T_dim; ++i)
                stream << separator << (*this)[i];
            stream << localeEnclosingEnd;
            return stream.str();
        }

        /** swizzle operator */
        template<typename T, T... T_values>
        constexpr auto operator[](Vec<T, sizeof...(T_values), detail::CVec<T, T_values...>> const v) const
        {
            using InType = ALPAKA_TYPEOF(v);
            return Vec<T_Type, InType::dim()>{(*this)[T_values]...};
        }

        template<typename T, T... T_values>
        constexpr auto ref(Vec<T, sizeof...(T_values), detail::CVec<T, T_values...>> const v)
        {
            using InType = ALPAKA_TYPEOF(v);
            using ArrayType = std::array<ALPAKA_TYPEOF(std::ref((*this)[T{0}])), sizeof...(T_values)>;
            auto array = ArrayType{std::ref((*this)[T_values])...};
            return Vec<T_Type, InType::dim(), ALPAKA_TYPEOF(array)>{array};
        }

        template<typename T, T... T_values>
        constexpr auto ref(Vec<T, sizeof...(T_values), detail::CVec<T, T_values...>> const v) const
        {
            using InType = ALPAKA_TYPEOF(v);
            using ArrayType = std::array<ALPAKA_TYPEOF(std::ref((*this)[T{0}])), sizeof...(T_values)>;
            auto array = ArrayType{std::ref((*this)[T_values])...};
            return Vec<T_Type, InType::dim(), ALPAKA_TYPEOF(array)>{array};
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

    private:
        /** reduce over a range of elements
         *
         * @tparam BinaryOp binary functor executed to reduce the range
         * @tparam T_start start index
         * @tparam T_end end index (excluded)
         * @return the type of the result depends on the binary functor
         */
        template<uint32_t T_start = 0u, uint32_t T_end = dim()>
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
    };

    template<std::size_t I, typename T_Type, uint32_t T_dim, typename T_Storage>
    constexpr auto get(Vec<T_Type, T_dim, T_Storage> const& v)
    {
        return v[I];
    }

    template<std::size_t I, typename T_Type, uint32_t T_dim, typename T_Storage>
    constexpr decltype(auto) get(Vec<T_Type, T_dim, T_Storage>& v)
    {
        return v[I];
    }

    template<typename T_Type>
    struct Vec<T_Type, 0>
    {
        using value_type = T_Type;
        static constexpr uint32_t dim = 0;

        template<typename T_OtherType>
        constexpr operator Vec<T_OtherType, 0>() const
        {
            return Vec<T_OtherType, 0>();
        }

        /**
         * == comparison operator.
         *
         * Returns always true
         */
        constexpr bool operator==(Vec const&) const
        {
            return true;
        }

        /**
         * != comparison operator.
         *
         * Returns always false
         */
        constexpr bool operator!=(Vec const&) const
        {
            return false;
        }

        static constexpr Vec create(T_Type)
        {
            /* this method should never be actually called,
             * it exists only for Visual Studio to handle alpaka::Size_t< 0 >
             */
            static_assert(sizeof(T_Type) != 0 && false);
        }
    };

    // type deduction guide
    template<typename T_1, typename... T_Args>
    ALPAKA_FN_DG Vec(T_1, T_Args...)
        -> Vec<T_1, uint32_t(sizeof...(T_Args) + 1u), ArrayStorage<T_1, uint32_t(sizeof...(T_Args) + 1u)>>;

    template<typename T_Type, uint32_t T_dim, typename T_Storage>
    std::ostream& operator<<(std::ostream& s, Vec<T_Type, T_dim, T_Storage> const& vec)
    {
        /* Usage of a temporary universal vector variable is required as workaround to silent an error with gcc 15
         * error: 'frameSpec' may be used uninitialized [-Werror=maybe-uninitialized] when compiling with 'ubsan'
         * enabled. It is a false positive in some cases where a CVec is used. Since this method is not performance
         * critical this workaround should be acceptable.
         */
        typename Vec<T_Type, T_dim, T_Storage>::UniVec v(vec);
        return s << v.toString();
    }

#define ALPAKA_VECTOR_BINARY_OP(typenameOrConcept, resultScalarType, op)                                              \
    template<typenameOrConcept T_Type, uint32_t T_dim, typename T_Storage, typename T_OtherStorage>                   \
    constexpr auto operator op(                                                                                       \
        const Vec<T_Type, T_dim, T_Storage>& lhs,                                                                     \
        const Vec<T_Type, T_dim, T_OtherStorage>& rhs)                                                                \
    {                                                                                                                 \
        /* to avoid allocation side effects the result is always a vector                                             \
         * with default policies                                                                                      \
         */                                                                                                           \
        Vec<resultScalarType, T_dim> result{};                                                                        \
        for(uint32_t i = 0u; i < T_dim; i++)                                                                          \
            result[i] = lhs[i] op rhs[i];                                                                             \
        return result;                                                                                                \
    }                                                                                                                 \
                                                                                                                      \
    template<                                                                                                         \
        typenameOrConcept T_Type,                                                                                     \
        concepts::LosslesslyConvertible<T_Type> T_ValueType,                                                          \
        uint32_t T_dim,                                                                                               \
        typename T_Storage>                                                                                           \
    constexpr auto operator op(const Vec<T_Type, T_dim, T_Storage>& lhs, T_ValueType rhs)                             \
    {                                                                                                                 \
        /* to avoid allocation side effects the result is always a vector                                             \
         * with default policies                                                                                      \
         */                                                                                                           \
        Vec<resultScalarType, T_dim> result{};                                                                        \
        for(uint32_t i = 0u; i < T_dim; i++)                                                                          \
            result[i] = lhs[i] op rhs;                                                                                \
        return result;                                                                                                \
    }                                                                                                                 \
    template<                                                                                                         \
        typenameOrConcept T_Type,                                                                                     \
        concepts::LosslesslyConvertible<T_Type> T_ValueType,                                                          \
        uint32_t T_dim,                                                                                               \
        typename T_Storage>                                                                                           \
    constexpr auto operator op(T_ValueType lhs, const Vec<T_Type, T_dim, T_Storage>& rhs)                             \
    {                                                                                                                 \
        /* to avoid allocation side effects the result is always a vector                                             \
         * with default policies                                                                                      \
         */                                                                                                           \
        Vec<resultScalarType, T_dim> result{};                                                                        \
        for(uint32_t i = 0u; i < T_dim; i++)                                                                          \
            result[i] = lhs op rhs[i];                                                                                \
        return result;                                                                                                \
    }

    /** binary operators
     * @{
     */
    ALPAKA_VECTOR_BINARY_OP(typename, T_Type, +)
    ALPAKA_VECTOR_BINARY_OP(typename, T_Type, -)
    ALPAKA_VECTOR_BINARY_OP(typename, T_Type, *)
    ALPAKA_VECTOR_BINARY_OP(typename, T_Type, /)
    ALPAKA_VECTOR_BINARY_OP(typename, bool, >=)
    ALPAKA_VECTOR_BINARY_OP(typename, bool, >)
    ALPAKA_VECTOR_BINARY_OP(typename, bool, <=)
    ALPAKA_VECTOR_BINARY_OP(typename, bool, <)
    ALPAKA_VECTOR_BINARY_OP(typename, bool, &&)
    ALPAKA_VECTOR_BINARY_OP(typename, bool, ||)
    ALPAKA_VECTOR_BINARY_OP(std::integral, T_Type, %)
    ALPAKA_VECTOR_BINARY_OP(std::integral, T_Type, <<)
    ALPAKA_VECTOR_BINARY_OP(std::integral, T_Type, >>)
    ALPAKA_VECTOR_BINARY_OP(std::integral, T_Type, &)
    ALPAKA_VECTOR_BINARY_OP(std::integral, T_Type, |)
    ALPAKA_VECTOR_BINARY_OP(std::integral, T_Type, ^)
    /** @} */

#undef ALPAKA_VECTOR_BINARY_OP

    /** Give the linear index of an N-dimensional index within an N-dimensional index space.
     *
     * @tparam T_IntegralType vector data type (must be an integral type)
     * @tparam T_dim dimension of the vector, should be >= 2
     * @param dim N-dimensional dim of the index space (N can be one dimension less compared to idx)
     * @param idx N-dimensional index within the index space
     *            @attention behaviour is undefined for negative index
     *            @attention if idx is outside of dim the result will be outside of the index domain too
     * @return linear index within the index domain
     *
     * @{
     */
    template<std::integral T_IntegralType, typename T_Storage, typename T_OtherStorage, uint32_t T_dim>
    constexpr T_IntegralType linearize(
        Vec<T_IntegralType, T_dim - 1u, T_Storage> const& dim,
        Vec<T_IntegralType, T_dim, T_OtherStorage> const& idx) requires(T_dim >= 2u)
    {
        T_IntegralType linearIdx{idx[0]};
        for(uint32_t d = 1u; d < T_dim; ++d)
            linearIdx = linearIdx * dim[d - 1u] + idx[d];

        return linearIdx;
    }

    template<std::integral T_IntegralType, typename T_Storage, typename T_OtherStorage, uint32_t T_dim>
    constexpr T_IntegralType linearize(
        Vec<T_IntegralType, T_dim, T_Storage> const& dim,
        Vec<T_IntegralType, T_dim, T_OtherStorage> const& idx)
    {
        return linearize(dim.template rshrink<T_dim - 1u>(), idx);
    }

    template<std::integral T_IntegralType, typename T_Storage, typename T_OtherStorage>
    ALPAKA_FN_HOST_ACC T_IntegralType linearize(
        Vec<T_IntegralType, 1u, T_Storage> const&,
        Vec<T_IntegralType, 1u, T_OtherStorage> const& idx)
    {
        return idx.x();
    }

    /** @} */

    /** Maps a linear index to an N-dimensional index
     *
     * @tparam T_IntegralType vector data type (must be an integral type)
     * @param dim N-dimensional index space
     * @param linearIdx Linear index within dim.
     *        @attention If linearIdx is an index outside of dim the result will be outside of the index domain
     * too.
     * @return N-dimensional index
     *
     * @{
     */
    template<std::integral T_IntegralType, typename T_Storage, uint32_t T_dim>
    constexpr Vec<T_IntegralType, T_dim> mapToND(
        Vec<T_IntegralType, T_dim, T_Storage> const& extents,
        T_IntegralType linearIdx) requires(T_dim >= 2u)
    {
        constexpr uint32_t reducedDim = T_dim - 1u;
        Vec<T_IntegralType, reducedDim> pitchExtents;
        pitchExtents.back() = extents.back();
        for(uint32_t d = 1u; d < T_dim - 1u; ++d)
            pitchExtents[reducedDim - 1u - d] = extents[T_dim - 1u - d] * pitchExtents[reducedDim - d];

        Vec<T_IntegralType, T_dim> result;
        for(uint32_t d = 0u; d < T_dim - 1u; ++d)
        {
            result[d] = linearIdx / pitchExtents[d];
            linearIdx -= pitchExtents[d] * result[d];
        }
        result[T_dim - 1u] = linearIdx;
        return result;
    }

    template<std::integral T_IntegralType, typename T_Storage>
    constexpr Vec<T_IntegralType, 1u> mapToND(
        Vec<T_IntegralType, 1u, T_Storage> const& extents,
        T_IntegralType linearIdx)
    {
        alpaka::unused(extents);
        return {linearIdx};
    }

    /** @} */

    namespace trait
    {
        template<typename T_Type, uint32_t T_dim, typename T_Storage>
        struct IsVector<Vec<T_Type, T_dim, T_Storage>> : std::true_type
        {
        };

        template<typename T_Type, uint32_t T_dim, T_Type... T_values>
        struct IsCVector<Vec<T_Type, T_dim, detail::CVec<T_Type, T_values...>>> : std::true_type
        {
        };
    } // namespace trait

    namespace trait
    {
        template<typename T_Type, uint32_t T_dim, typename T_Storage>
        struct GetDim<alpaka::Vec<T_Type, T_dim, T_Storage>>
        {
            static constexpr uint32_t value = T_dim;
        };

        template<typename T>
        struct GetVec;

        template<std::integral T>
        struct GetVec<T>
        {
            using type = Vec<T, 1u>;
        };

        template<typename T_Type, uint32_t T_dim, typename T_Storage>
        struct GetVec<alpaka::Vec<T_Type, T_dim, T_Storage>>
        {
            using type = alpaka::Vec<T_Type, T_dim, T_Storage>;
        };

        template<typename T>
        using GetVec_t = typename GetVec<T>::type;

        template<typename T_Type, uint32_t T_dim, typename T_Storage>
        struct GetValueType<Vec<T_Type, T_dim, T_Storage>>
        {
            using type = T_Type;
        };

    } // namespace trait

    template<typename T>
    consteval auto getVec(T const& any)
    {
        return trait::GetVec_t<T>{any};
    }

    namespace internal
    {
        template<typename T_To, typename T_Type, uint32_t T_dim, typename T_Storage>
        struct PCast::Op<T_To, alpaka::Vec<T_Type, T_dim, T_Storage>>
        {
            constexpr auto operator()(auto&& input) const
                requires std::convertible_to<T_Type, T_To> && (!std::same_as<T_To, T_Type>)
            {
                return typename alpaka::Vec<T_To, T_dim, T_Storage>::UniVec([&](uint32_t idx) constexpr
                                                                            { return static_cast<T_To>(input[idx]); });
            }

            constexpr decltype(auto) operator()(auto&& input) const requires std::same_as<T_To, T_Type>
            {
                return std::forward<decltype(input)>(input);
            }
        };
    } // namespace internal

    /** @todo the function for integral values is defined in Utils.hpp
     * move this to a better place, e.g. math and expose this for the user too
     */
    template<concepts::Vector T_Vector0, concepts::Vector T_Vector1>
    requires(std::is_same_v<GetValueType_t<T_Vector0>, GetValueType_t<T_Vector1>>)
    [[nodiscard]] ALPAKA_FN_HOST_ACC constexpr concepts::Vector auto divCeil(T_Vector0 a, T_Vector1 b)
    {
        return (a + b - T_Vector0::fill(1)) / b;
    }

    template<concepts::Vector T_Vector0, concepts::Vector T_Vector1>
    requires(std::is_same_v<GetValueType_t<T_Vector0>, GetValueType_t<T_Vector1>>)
    [[nodiscard]] ALPAKA_FN_HOST_ACC constexpr concepts::Vector auto divExZero(T_Vector0 a, T_Vector1 b)
    {
        auto tmp = a / b;

        using ValueType = alpaka::GetValueType_t<T_Vector0>;
        for(uint32_t d = 0u; d < a.dim(); ++d)
            tmp[d] = std::max(tmp[d], ValueType{1u});
        return tmp;
    }
}; // namespace alpaka

namespace std
{
    template<typename T_Type, uint32_t T_dim, typename T_Storage>
    struct tuple_size<alpaka::Vec<T_Type, T_dim, T_Storage>>
    {
        static constexpr std::size_t value = T_dim;
    };

    template<std::size_t I, typename T_Type, uint32_t T_dim, typename T_Storage>
    struct tuple_element<I, alpaka::Vec<T_Type, T_dim, T_Storage>>
    {
        using type = T_Type;
    };
} // namespace std
