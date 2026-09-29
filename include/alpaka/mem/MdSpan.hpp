/* Copyright 2025 René Widera, Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/interface.hpp"
#include "alpaka/mem/Alignment.hpp"
#include "alpaka/mem/DataPitches.hpp"
#include "alpaka/mem/MdForwardIter.hpp"
#include "alpaka/mem/concepts/detail/InnerTypeAllowedCast.hpp"
#include "alpaka/mem/trait.hpp"
#include "alpaka/onHost/interface.hpp"
#include "alpaka/trait.hpp"
#include "concepts/IndexVec.hpp"

#include <type_traits>

namespace alpaka
{
    /** Lightweight view to data in an n-dimensional array.
     *
     * Const-ness of the MdSpan instance is propagated to the data region.
     * A constant MdSpan can be used to access non-const data.
     *
     * @tparam T_Type if the type is const the data is only readable
     */
    template<
        typename T_Type,
        concepts::Vector T_Extents,
        concepts::Vector T_Pitches,
        concepts::Alignment T_MemAlignment = Alignment<>>
    struct MdSpan;

    template<concepts::Alignment T_MemAlignment = Alignment<>>
    inline constexpr auto makeMdSpan(
        auto* pointer,
        concepts::Vector auto const& extents,
        concepts::Vector auto const& pitchBytes,
        T_MemAlignment const memAlignment = T_MemAlignment{})
    {
        return MdSpan{pointer, extents, pitchBytes, memAlignment};
    }

    template<typename T_ValueType, concepts::Alignment T_MemAlignment = Alignment<>>
    inline constexpr auto makeMdSpan(
        T_ValueType* pointer,
        concepts::Vector auto const& extents,
        T_MemAlignment const memAlignment = T_MemAlignment{})
    {
        auto pitchMd = alpaka::calculatePitchesFromExtents<T_ValueType>(extents);
        return MdSpan{pointer, extents, pitchMd, memAlignment};
    }

    inline constexpr auto makeMdSpan(auto&& any)
    {
        return MdSpan{onHost::data(any), onHost::getExtents(any), onHost::getPitches(any), alpaka::getAlignment(any)};
    }

    template<
        typename T_Type,
        concepts::Vector T_Extents,
        concepts::Vector T_Pitches,
        concepts::Alignment T_MemAlignment>
    struct MdSpan
    {
        using value_type = T_Type;
        using reference = value_type&;
        using const_reference = std::add_const_t<value_type>&;
        using pointer = value_type*;
        using const_pointer = std::add_const_t<value_type>*;
        using index_type = typename T_Pitches::value_type;

        using ConstThis = MdSpan<std::add_const_t<value_type>, T_Extents, T_Pitches, T_MemAlignment>;

        static_assert(std::is_convertible_v<index_type, typename T_Extents::value_type>);
        static_assert(T_Extents::dim() == T_Pitches::dim());

        static consteval uint32_t dim()
        {
            return T_Extents::dim();
        }

        /** return value the origin pointer is pointing to
         *
         * @return value at the current location
         */
        constexpr const_reference operator*() const
        {
            return *this->m_ptr;
        }

        constexpr reference operator*()
        {
            return *this->m_ptr;
        }

        /** get origin pointer
         *
         * If the pointer is const and therefore read only depends on T_Type and not the const-ness of MdSpan.
         */
        constexpr const_pointer data() const
        {
            return this->m_ptr;
        }

        constexpr pointer data()
        {
            return this->m_ptr;
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

        constexpr auto cbegin() const
        {
            return MdForwardIter{this->getConstMdSpan()};
        }

        constexpr auto cend() const
        {
            return MdForwardIterEnd{this->getConstMdSpan()};
        }

        /*Object must init by copy a valid instance*/
        constexpr MdSpan() = default;

        /** Constructor
         *
         * @param pointer pointer to the memory
         * @param extents number of elements
         * @param pitchBytes pitch in bytes per dimension
         * @param memAlignmentInByte alignment in bytes (zero will set alignment to element alignment)
         */
        constexpr MdSpan(
            T_Type* pointer,
            T_Extents extents,
            T_Pitches const& pitchBytes,
            [[maybe_unused]] T_MemAlignment const& memAlignmentInByte = T_MemAlignment{})
            : m_ptr(pointer)
            , m_extent(extents)
            , m_pitch(pitchBytes)
        {
        }

        template<typename T_TypeOther>
        requires internal::concepts::InnerTypeAllowedCast<T_Type, T_TypeOther>
        constexpr MdSpan(MdSpan<T_TypeOther, T_Extents, T_Pitches, T_MemAlignment> const& other)
            : m_ptr(other.data())
            , m_extent(other.getExtents())
            , m_pitch(other.getPitches())
        {
        }

        template<typename T_TypeOther>
        requires alpaka::internal::concepts::InnerTypeAllowedCast<T_Type, T_TypeOther>
        constexpr MdSpan(MdSpan<T_TypeOther, T_Extents, T_Pitches, T_MemAlignment>&& other)
            : m_ptr(std::move(other.data()))
            , m_extent(std::move(other.getExtents()))
            , m_pitch(std::move(other.getPitches()))
        {
        }

        constexpr MdSpan(MdSpan const&) = default;
        constexpr MdSpan(MdSpan&&) = default;

        /** Assignment operator keeping const-ness
         *
         * @attention the assign operator is not removing inner const-ness because the type signature is not changed.
         */
        constexpr MdSpan& operator=(MdSpan const&) = default;

        constexpr MdSpan& operator=(MdSpan&&) = default;

        static constexpr auto getAlignment()
        {
            return T_MemAlignment{};
        }

        /** get value at the given index
         *
         * @param idx n-dimensional offset, relative to the origin pointer
         * @return reference to the value
         */
        constexpr const_reference operator[](
            // cannot use dim() or alpaka::trait::GetDim_v<T_Extents> because they cause a segmentation fault in nvcc
            concepts::IndexVec<index_type, alpaka::trait::GetDim<T_Extents>::value> auto const& idx) const
        {
            return *ptr(idx);
        }

        constexpr reference operator[](
            // cannot use dim() or alpaka::trait::GetDim_v<T_Extents> because they cause a segmentation fault in nvcc
            concepts::IndexVec<index_type, alpaka::trait::GetDim<T_Extents>::value> auto const& idx)
        {
            return *ptr(idx);
        }

        constexpr const_reference operator[](std::integral auto const& idx) const requires(dim() == 1u)
        {
            return *ptr(Vec{idx});
        }

        constexpr reference operator[](std::integral auto const& idx) requires(dim() == 1u)
        {
            return *ptr(Vec{idx});
        }

        constexpr auto getExtents() const
        {
            return m_extent;
        }

        constexpr T_Pitches getPitches() const
        {
            return m_pitch.getPitches();
        }

        constexpr auto getConstMdSpan() const
        {
            using ConstValueType = std::add_const_t<value_type>;
            return makeMdSpan(
                static_cast<ConstValueType*>(m_ptr),
                this->getExtents(),
                this->getPitches(),
                T_MemAlignment{});
        }

        /** True if MdSpan is pointing to valid memory.
         *
         * @details
         * An MdSpan remains valid even after being moved. The reason for this is that the MdSpan is simply copied.
         * This is more efficient than a real move (e.g., setting the data pointer to nullptr). Implementing a real
         * move is also not possible because MdSpan must be trivially copyable, which requires a default move
         * constructor.
         */
        [[nodiscard]] constexpr explicit operator bool() const noexcept
        {
            return true;
        }

    protected:
        /** get the pointer of the value relative to the origin pointer m_ptr
         *
         * @param idx n-dimensional offset
         * @return pointer to value
         */
        constexpr auto ptr(concepts::Vector auto const& idx) const requires(dim() >= 2u)
        {
            /** offset in bytes
             *
             * We calculate the complete offset in bytes even if it would be possible to change the x-dimension
             * with the native value_types pointer, this is reducing the register footprint.
             */
            index_type offset = sizeof(value_type) * idx.back();
            for(uint32_t d = 0u; d < dim() - 1u; ++d)
            {
                offset += m_pitch[d] * idx[d];
            }
            using CharPtrType = std::conditional_t<std::is_const_v<value_type>, char const*, char*>;
            using ResultPtrType = std::conditional_t<std::is_const_v<value_type>, const_pointer, pointer>;
            return reinterpret_cast<ResultPtrType>(reinterpret_cast<CharPtrType>(this->m_ptr) + offset);
        }

        constexpr const_pointer ptr(concepts::Vector auto const& idx) const requires(dim() == 1u)
        {
            return this->m_ptr + idx.x();
        }

        constexpr pointer ptr(concepts::Vector auto const& idx) requires(dim() == 1u)
        {
            return this->m_ptr + idx.x();
        }

    private:
        pointer m_ptr;
        T_Extents m_extent;
        DataPitches<value_type, T_Pitches> m_pitch;
    };

    template<
        typename T_Type,
        concepts::Vector T_Extents,
        concepts::Vector T_Pitches,
        concepts::Alignment T_MemAlignment>
    std::ostream& operator<<(std::ostream& s, MdSpan<T_Type, T_Extents, T_Pitches, T_MemAlignment> const& mdSpan)
    {
        return s << "MdSpan{ dim=" << ALPAKA_TYPEOF(mdSpan)::dim() << ", extents=" << mdSpan.getExtents().toString()
                 << ", pitches=" << mdSpan.getPitches().toString()
                 << " , alignment=" << T_MemAlignment::template get<T_Type>() << " }";
    }

    template<
        typename T_Type,
        alpaka::concepts::Vector T_Extents,
        alpaka::concepts::Vector T_Pitches,
        alpaka::concepts::Alignment T_MemAlignment>
    struct internal::CopyConstructableDataSource<MdSpan<T_Type, T_Extents, T_Pitches, T_MemAlignment>> : std::true_type
    {
        using InnerMutable = MdSpan<std::remove_const_t<T_Type>, T_Extents, T_Pitches, T_MemAlignment>;
        using InnerConst = MdSpan<std::add_const_t<T_Type>, T_Extents, T_Pitches, T_MemAlignment>;
    };
} // namespace alpaka
