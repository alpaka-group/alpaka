/* Copyright 2025 René Widera, Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/CVec.hpp"
#include "alpaka/Vec.hpp"
#include "alpaka/concepts/types.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/mem/Alignment.hpp"
#include "alpaka/mem/DataPitches.hpp"
#include "alpaka/mem/MdForwardIter.hpp"
#include "alpaka/mem/concepts/detail/InnerTypeAllowedCast.hpp"
#include "alpaka/mem/trait.hpp"
#include "alpaka/trait.hpp"
#include "concepts/IndexVec.hpp"

#include <concepts>
#include <type_traits>

namespace alpaka
{
    /** access a C array with compile time extents via a runtime md index. */
    template<std::integral auto T_numDims, uint32_t T_dim = 0u>
    struct ResolveArrayAccess
    {
        constexpr decltype(auto) operator()(auto arrayPtr, concepts::Vector auto const& idx) const
        {
            return ResolveArrayAccess<T_numDims - 1u, T_dim + 1u>{}(arrayPtr[idx[T_dim]], idx);
        }
    };

    template<uint32_t T_dim>
    struct ResolveArrayAccess<1u, T_dim>
    {
        constexpr decltype(auto) operator()(auto arrayPtr, concepts::Vector auto const& idx) const
        {
            return arrayPtr[idx[T_dim]];
        }
    };

    /** build C array type with compile time extents from a scalar value based on the compile time extents vector */
    template<typename T, concepts::CVector T_Extents, uint32_t T_numDims = T_Extents::dim(), uint32_t T_dim = 0u>
    struct CArrayType
    {
        using type =
            typename CArrayType<T[T_Extents{}[T_numDims - T_dim - 1u]], T_Extents, T_numDims - 1u, T_dim + 1u>::type;
    };

    template<typename T, concepts::CVector T_Extents, uint32_t T_dim>
    struct CArrayType<T, T_Extents, 1u, T_dim>
    {
        using type = T[T_Extents{}[0u]];
    };

    template<typename T_ArrayType, std::integral T_IndexType, concepts::Alignment T_MemAlignment = Alignment<>>
    struct MdSpanArray
    {
        static_assert(
            sizeof(T_ArrayType) && false,
            "MdSpanArray can only be used if std::is_array_v<T> is true for the given type.");
    };

    template<alpaka::concepts::CStaticArray T_ArrayType, std::integral T_IndexType, concepts::Alignment T_MemAlignment>
    struct MdSpanArray<T_ArrayType, T_IndexType, T_MemAlignment>
    {
    private:
        using MutArrayType = std::remove_cv_t<T_ArrayType>;
        using ConstArrayType = std::add_const_t<MutArrayType>;

    public:
        using value_type = std::remove_all_extents_t<T_ArrayType>;
        using reference = value_type&;
        using const_reference = value_type const&;
        using pointer = value_type*;
        using const_pointer = value_type const*;
        using index_type = T_IndexType;

        static consteval uint32_t dim()
        {
            return std::rank_v<T_ArrayType>;
        }

        /** return value the origin pointer is pointing to
         *
         * @return value at the current location
         */
        constexpr const_reference operator*() const
        {
            return *this->data();
        }

        constexpr reference operator*()
        {
            return *this->data();
        }

        /** get origin pointer */
        constexpr const_pointer data() const
        {
            return reinterpret_cast<const_pointer>(this->m_ptr);
        }

        constexpr pointer data()
        {
            return reinterpret_cast<pointer>(this->m_ptr);
        }

        constexpr auto begin() const
        {
            return MdForwardIter{this->getConstMdSpan()};
        }

        constexpr auto begin()
        {
            return MdForwardIter{*this};
        }

        constexpr auto end() const
        {
            return MdForwardIterEnd{this->getConstMdSpan()};
        }

        constexpr auto end()
        {
            return MdForwardIterEnd{*this};
        }

        constexpr auto getConstMdSpan() const
        {
            return MdSpanArray<ConstArrayType, T_IndexType, T_MemAlignment>(*m_ptr);
        }

        constexpr auto cbegin() const
        {
            return MdForwardIter{this->getConstMdSpan()};
        }

        constexpr auto cend() const
        {
            return MdForwardIterEnd{this->getConstMdSpan()};
        }

        /*Object must init by copy a valid instance*/
        constexpr MdSpanArray() = default;

        /** Constructor
         *
         * @param pointer pointer to the memory
         */
        constexpr MdSpanArray(T_ArrayType& staticSizedArray) : m_ptr(const_cast<MutArrayType*>(&staticSizedArray))
        {
        }

        template<alpaka::concepts::CStaticArray T_OtherArrayType>
        requires internal::concepts::InnerTypeAllowedCast<T_ArrayType, T_OtherArrayType>
        constexpr MdSpanArray(MdSpanArray<T_OtherArrayType, T_IndexType, T_MemAlignment> const& other)
            : m_ptr(other.m_ptr)
        {
        }

        constexpr MdSpanArray(MdSpanArray const&) = default;

        /** Assignment operator keeping const-ness
         *
         * alpaka keeps track that const references to a span can not be cast to non const.
         * If the inner value type is not const the second assignment operator with non const will be used.
         */
        constexpr MdSpanArray& operator=(MdSpanArray const& other)
            requires(std::is_const_v<typename ALPAKA_TYPEOF(other)::value_type>)
            = default;
        constexpr MdSpanArray& operator=(MdSpanArray&) = default;

        constexpr MdSpanArray(MdSpanArray&&) = default;

        template<alpaka::concepts::CStaticArray T_OtherArrayType>
        requires internal::concepts::InnerTypeAllowedCast<T_ArrayType, T_OtherArrayType>
        constexpr MdSpanArray(MdSpanArray<T_OtherArrayType, T_IndexType, T_MemAlignment>&& other) : m_ptr(other.m_ptr)
        {
        }

        constexpr MdSpanArray& operator=(MdSpanArray&&) = default;

        static constexpr auto getAlignment()
        {
            return T_MemAlignment{};
        }

        /** get value at the given index
         *
         * @param idx offset relative to the origin pointer
         * @return reference to the value
         * @{
         */
        constexpr const_reference operator[](
            // cannot use dim() or std::rank_v<T_ArrayType> because the cause a segmentation fault in nvcc
            concepts::IndexVec<index_type, std::rank<T_ArrayType>::value> auto const& idx) const
        {
            return ResolveArrayAccess<dim()>{}(*m_ptr, idx);
        }

        constexpr reference operator[](
            // cannot use dim() or std::rank_v<T_ArrayType> because the cause a segmentation fault in nvcc
            concepts::IndexVec<index_type, std::rank<T_ArrayType>::value> auto const& idx)
        {
            return ResolveArrayAccess<dim()>{}(*m_ptr, idx);
        }

        constexpr const_reference operator[](index_type const& idx) const
        {
            return (*m_ptr)[idx];
        }

        constexpr reference operator[](index_type const& idx)
        {
            return (*m_ptr)[idx];
        }

        constexpr bool operator==(MdSpanArray const other) const
        {
            return m_ptr == other.m_ptr;
        }

        /** @} */

        constexpr auto getExtents() const
        {
            // uint32_t is the data type of dim()
            auto const createExtents = []<uint32_t... T_extent>(std::integer_sequence<uint32_t, T_extent...>)
            { return CVec<index_type, std::extent_v<T_ArrayType, T_extent>...>{}; };
            return createExtents(std::make_integer_sequence<uint32_t, dim()>{});
        }

        constexpr auto getPitches() const
        {
            return alpaka::calculatePitchesFromExtents<value_type>(getExtents());
        }

        /** True if MdSpanArray is pointing to valid memory.
         *
         * @details
         * An MdSpanArray remains valid even after being moved. The reason is, that it uses stack memory which cannot
         * be freed.
         */
        [[nodiscard]] constexpr explicit operator bool() const noexcept
        {
            return true;
        }

        // Needs to be friend of itself with that the copy and move constructor can access the m_ptr of other, if the
        // const modifier of the C static array type of the other type is different.
        friend MdSpanArray<MutArrayType, T_IndexType, T_MemAlignment>;
        friend MdSpanArray<ConstArrayType, T_IndexType, T_MemAlignment>;

    protected:
        // we store the C static array as mutable type that we can assign it another MdSpanArray with const or
        // non-const inner type
        // Depending on the value_type, the const is added at memory access
        MutArrayType* m_ptr;
    };

    template<
        alpaka::concepts::CStaticArray T_ArrayType,
        std::integral T_IndexType,
        alpaka::concepts::Alignment T_MemAlignment>
    struct internal::CopyConstructableDataSource<MdSpanArray<T_ArrayType, T_IndexType, T_MemAlignment>>
        : std::true_type
    {
        using InnerMutable = MdSpanArray<std::remove_const_t<T_ArrayType>, T_IndexType, T_MemAlignment>;
        using InnerConst = MdSpanArray<std::add_const_t<T_ArrayType>, T_IndexType, T_MemAlignment>;
    };
} // namespace alpaka
