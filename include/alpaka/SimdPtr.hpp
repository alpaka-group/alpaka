/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Simd.hpp"
#include "alpaka/Vec.hpp"
#include "alpaka/internal/interface.hpp"
#include "alpaka/mem/Alignment.hpp"
#include "alpaka/mem/concepts/IMdSpan.hpp"
#include "alpaka/mem/concepts/IndexVec.hpp"
#include "alpaka/trait.hpp"

#include <concepts>
#include <cstdint>
#include <type_traits>

namespace alpaka
{
    namespace trait
    {
        template<typename T>
        struct IsSimdPtr : std::false_type
        {
        };
    } // namespace trait

    template<typename T>
    constexpr bool isSimdPtr_v = trait::IsSimdPtr<T>::value;

    namespace concepts
    {
        /** Concept to check if a type is a SIMD pointer
         *
         * @tparam T Type to check
         * @tparam T_ValueType enforce a value type of the SIMD pointer, if not provided the value type is not checked
         * @tparam T_width enforce lane width of the SIMD pointer, if not provided the value is not checked
         */
        template<typename T, typename T_ValueType = alpaka::NotRequired, uint32_t T_width = alpaka::notRequiredWidth>
        concept SimdPtr = isSimdPtr_v<T>
                          && (std::same_as<T_ValueType, GetValueType_t<std::decay_t<T>>>
                              || std::same_as<T_ValueType, alpaka::NotRequired>)
                          && ((T_width == alpaka::notRequiredWidth) || (T::width() == T_width));
    } // namespace concepts

    /** pointer to a SIMD pack with the width T_SimdWidth
     *
     * The pointer is used to load/store data from/to memory
     *
     * @tparam T_MdSpan type of the memory the pointer is pointing to
     * @tparam T_IdxType type of the index
     * @tparam T_MemAlignment alignment of the memory the pointer is pointing to
     * @tparam T_SimdWidth width of the SIMD pack
     */
    template<
        typename T_MdSpan,
        alpaka::concepts::Vector T_IdxType,
        alpaka::concepts::Alignment T_MemAlignment,
        alpaka::concepts::CVector T_SimdWidth>
    struct SimdPtr : private T_MdSpan
    {
        using value_type = typename T_MdSpan::value_type;
        using IdxType = typename T_IdxType::UniVec;

        static consteval uint32_t width()
        {
            return T_SimdWidth{}.back();
        }

        constexpr SimdPtr(T_MdSpan const& mdSpan, T_IdxType const& idx, T_MemAlignment, T_SimdWidth)
            : T_MdSpan(mdSpan)
            , m_idx(idx)
        {
        }

        /** Shift the element the pointer is pointing to by idx
         *
         * @param idx number of elements to shift the pointer by
         * @return a new simd pointer pointing to the shifted element
         *
         * @{
         */
        constexpr alpaka::concepts::SimdPtr auto operator[](auto const& idx) const
        {
            /* Do not use concepts::IndexVec as concept in the function signature else nvcc (tested 12.X -> 13.0)
             * segfaults during compile.
             */
            static_assert(
                alpaka::concepts::IndexVec<ALPAKA_TYPEOF(idx), typename IdxType::value_type, T_MdSpan::dim()>,
                "The dimension of idx must match the encapsulated MdSpan dimension and the index type of idx must be "
                "lossless castable to the MdSpan index type");
            constexpr uint32_t valueAlignment = static_cast<uint32_t>(alignof(value_type));
            constexpr auto align = Alignment<valueAlignment>{};
            return SimdPtr<T_MdSpan, IdxType, ALPAKA_TYPEOF(align), T_SimdWidth>{
                static_cast<T_MdSpan>(*this),
                idx + m_idx,
                align,
                T_SimdWidth{}};
        }

        constexpr alpaka::concepts::SimdPtr auto operator[](auto const& idx)
        {
            /* Do not use concepts::IndexVec as concept in the function signature else nvcc (tested 12.X -> 13.0)
             * segfaults during compile.
             */
            static_assert(
                alpaka::concepts::IndexVec<ALPAKA_TYPEOF(idx), typename IdxType::value_type, T_MdSpan::dim()>,
                "The dimension of idx must match the encapsulated MdSpan dimension and the index type of idx must be "
                "lossless castable to the MdSpan index type");
            constexpr uint32_t valueAlignment = static_cast<uint32_t>(alignof(value_type));
            constexpr auto align = Alignment<valueAlignment>{};
            return SimdPtr<T_MdSpan, IdxType, ALPAKA_TYPEOF(align), T_SimdWidth>{
                static_cast<T_MdSpan>(*this),
                idx + m_idx,
                align,
                T_SimdWidth{}};
        }

        /** @} */

        constexpr decltype(auto) load() const
        {
            return internal::loadAsSimd<width()>(static_cast<T_MdSpan const&>(*this), getAlignment(), m_idx);
        }

        constexpr decltype(auto) load()
        {
            return internal::loadAsSimd<width()>(static_cast<T_MdSpan&>(*this), getAlignment(), m_idx);
        }

        /** get the alignment of the memory the pointer is pointing to
         *
         * @attention If the pointer is shifted by `operator[]` the alignment is equal to the data alignment of an
         * single element
         *
         * @return the alignment of the memory (in byte) the pointer is pointing to
         */
        static constexpr auto getAlignment()
        {
            using SpanElemType = typename T_MdSpan::value_type;
            constexpr uint32_t spanAlignment = T_MdSpan::getAlignment().template get<SpanElemType>();
            using MemoryAlignment = std::conditional_t<
                std::is_same_v<AutoAligned, T_MemAlignment>,
                Alignment<spanAlignment>,
                Alignment<std::min(T_MemAlignment::template get<SpanElemType>(), spanAlignment)>>;
            return MemoryAlignment{};
        }

        /** store the simd pack to the memory the pointer is pointing to
         *
         * @param rhs simd pack to store
         *
         * @{
         */
        template<typename T_Storage>
        constexpr void storeTo(Simd<value_type, SimdPtr::width(), T_Storage> const& rhs) const
        {
            auto* ptr = &T_MdSpan::operator[](m_idx);

            rhs.copyTo(ptr, getAlignment());
        }

        template<typename T_Storage>
        constexpr void storeTo(Simd<value_type, SimdPtr::width(), T_Storage> const& rhs)
        {
            auto* ptr = &T_MdSpan::operator[](m_idx);
            rhs.copyTo(ptr, getAlignment());
        }

        template<typename T_Storage>
        constexpr SimdPtr const& operator=(Simd<value_type, SimdPtr::width(), T_Storage> const& rhs) const
        {
            storeTo(rhs);
            return *this;
        }

        template<typename T_Storage>
        constexpr SimdPtr& operator=(Simd<value_type, SimdPtr::width(), T_Storage> const& rhs)
        {
            storeTo(rhs);
            return *this;
        }

        /** @} */

        /** offset in elements relative to the MdSpan given at construction
         *
         * The index points to the first element followed by T_SimdWidth elements.
         *
         * @return the index of the first element relative to the MdSpan given at construction
         */
        constexpr IdxType getIdx() const
        {
            return m_idx;
        }

    private:
        IdxType m_idx;
    };

    namespace internal
    {
        template<
            alpaka::concepts::IMdSpan T_MdSpan,
            alpaka::concepts::Alignment T_MdSpanAlignment,
            alpaka::concepts::Vector T_Idx>
        struct LoadAsSimd::Op<T_MdSpan, T_MdSpanAlignment, T_Idx>
        {
            template<uint32_t T_simdWidth>
            constexpr auto load(auto&& dataSource, T_MdSpanAlignment alignment, T_Idx const& idx) const
            {
                static_assert(
                    std::is_same_v<T_MdSpan, ALPAKA_TYPEOF(dataSource)>,
                    "Data source type must match the class template signature.");
                auto&& d = dataSource[idx];
                using DataTypeType = std::remove_reference_t<decltype(d)>;
                using DstType = std::conditional_t<
                    std::is_const_v<DataTypeType>,
                    Simd<std::decay_t<DataTypeType>, T_simdWidth> const,
                    Simd<std::decay_t<DataTypeType>, T_simdWidth>>;

                std::remove_const_t<DstType> dest;
                dest.copyFrom(&d, alignment);
                return dest;
            }
        };
    } // namespace internal

    namespace trait
    {
        template<typename T>
        requires(isSpecializationOf_v<T, SimdPtr>)
        struct IsSimdPtr<T> : std::true_type
        {
        };
    } // namespace trait
} // namespace alpaka
