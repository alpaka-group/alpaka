/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/core/PP.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/mem/BoundaryIter.hpp"
#include "alpaka/mem/FlatIdxContainer.hpp"

#include <cstdint>

namespace alpaka
{

    template<
        concepts::VectorOrScalar T_End,
        concepts::Vector T_Begin = typename T_End::UniVec,
        concepts::Vector T_Stride = typename T_End::UniVec>
    struct IdxRange
    {
        using IdxType = typename T_End::value_type;
        using IdxVecType = typename T_End::UniVec;

        constexpr IdxRange(T_Begin const& begin, T_End const& end, T_Stride const& stride)
            : m_begin{begin}
            , m_end{end}
            , m_stride{stride}
        {
        }

        constexpr IdxRange(T_Begin const& begin, T_End const& end)
            : m_begin{begin}
            , m_end{end}
            , m_stride{T_End::fill(1u)}
        {
        }

        constexpr IdxRange(T_End const& extent) : m_begin{T_End::fill(0u)}, m_end{extent}, m_stride{T_End::fill(1u)}
        {
        }

        static consteval uint32_t dim()
        {
            return IdxVecType::dim();
        }

        template<concepts::TypeOrVector<typename T_End::value_type> T_OpType>
        ALPAKA_FN_HOST_ACC constexpr auto operator%(T_OpType const& rhs) const
        {
            return IdxRange<T_End, T_Begin, ALPAKA_TYPEOF(m_stride * rhs)>{m_begin, m_end, m_stride * rhs};
        }

        template<concepts::TypeOrVector<typename T_End::value_type> T_OpType>
        ALPAKA_FN_HOST_ACC constexpr auto operator>>(T_OpType const& rhs) const
        {
            return IdxRange<ALPAKA_TYPEOF(m_end + rhs), ALPAKA_TYPEOF(m_begin + rhs), ALPAKA_TYPEOF(m_stride)>{
                m_begin + rhs,
                m_end + rhs,
                m_stride};
        }

        template<concepts::TypeOrVector<typename T_End::value_type> T_OpType>
        ALPAKA_FN_HOST_ACC constexpr auto operator<<(T_OpType const& rhs) const
        {
            return IdxRange<ALPAKA_TYPEOF(m_end - rhs), ALPAKA_TYPEOF(m_begin - rhs), T_Stride>{
                m_begin - rhs,
                m_end - rhs,
                m_stride};
        }

        constexpr auto distance() const
        {
            return m_end - m_begin;
        }

        /** Begin iterator to iterate all positions in the range. It first iterates the fastest index (the one on the
         * far right -> x-dimension) and then moves sequentially to the slowest index (the one on the far left) until
         * it reaches the end.
         *
         * If you want to iterate the index in parallel with many threads, use the function
         * alpaka::onAcc::makeIdxMap().
         *
         * @return Begin iterator
         */
        [[nodiscard]] constexpr auto begin() const
        {
            return alpaka::onAcc::FlatIdxContainer{
                *this,
                alpaka::ThreadSpace{T_End::fill(0), T_End::fill(1)},
                alpaka::onAcc::layout::contiguous,
                alpaka::iotaCVec<typename ALPAKA_TYPEOF(distance())::value_type, ALPAKA_TYPEOF(distance())::dim()>()}
                .begin();
        }

        [[nodiscard]] constexpr auto end() const
        {
            return alpaka::onAcc::FlatIdxContainer{
                *this,
                alpaka::ThreadSpace{T_End::fill(0), T_End::fill(1)},
                alpaka::onAcc::layout::contiguous,
                alpaka::iotaCVec<typename ALPAKA_TYPEOF(distance())::value_type, ALPAKA_TYPEOF(distance())::dim()>()}
                .end();
        }

        std::string toString(std::string const& separator = ",", std::string const& enclosings = "{}") const
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
            stream << localeEnclosingBegin;
            stream << m_begin << separator << m_end << separator << m_stride;
            stream << localeEnclosingEnd;
            return stream.str();
        }

        /** Return the begin position of the range as vector coordinate.
         * Allows not to iterate of the IdxRange. Use begin() instead of. **/
        [[nodiscard]] constexpr auto getBeginMd() const -> T_Begin
        {
            return m_begin;
        }

        /** Return the end position of the range as vector coordinate.
         * Allows not to iterate of the IdxRange. Use end() instead of. **/
        [[nodiscard]] constexpr auto getEndMd() const -> T_End
        {
            return m_end;
        }

        /** Return the stride of the range. **/
        [[nodiscard]] constexpr auto getStrideMd() const -> T_Stride
        {
            return m_stride;
        }

        using value_type = typename T_Begin::value_type;

    private:
        T_Begin m_begin;
        T_End m_end;
        T_Stride m_stride;
    };

    template<uint32_t T_dim, alpaka::concepts::Vector T_LowHaloVec, alpaka::concepts::Vector T_UpHaloVec>
    constexpr auto makeDirectionSubRange(
        auto const range,
        alpaka::BoundaryDirection<T_dim, T_LowHaloVec, T_UpHaloVec> const& boundaryDir)
    {
        auto begin = Vec<uint32_t, T_dim>::fill(0u);
        auto end = Vec<uint32_t, T_dim>::fill(0u);
        for(uint32_t i = 0; i < T_dim; ++i)
        {
            switch(boundaryDir.data[i])
            {
            case BoundaryType::lower:
                begin[i] = range.getBeginMd()[i];
                end[i] = range.getBeginMd()[i] + boundaryDir.lowerHaloSize[i];
                break;
            case BoundaryType::upper:
                begin[i] = range.getEndMd()[i] - boundaryDir.upperHaloSize[i];
                end[i] = range.getEndMd()[i];
                break;
            case BoundaryType::middle:
                begin[i] = range.getBeginMd()[i] + boundaryDir.lowerHaloSize[i];
                end[i] = range.getEndMd()[i] - boundaryDir.upperHaloSize[i];
                break;
            case BoundaryType::oob:
                [[fallthrough]];
            default:
                ALPAKA_ASSERT_ACC(false);
            }
        }
        return IdxRange{begin, end, range.getStrideMd()};
    }

    namespace internal
    {
        template<
            typename T_To,
            alpaka::concepts::Vector T_End,
            alpaka::concepts::Vector T_Begin,
            alpaka::concepts::Vector T_Stride>
        struct PCast::Op<T_To, IdxRange<T_End, T_Begin, T_Stride>>
        {
            constexpr auto operator()(auto&& input) const
                requires std::convertible_to<typename T_End::value_type, T_To>
                         && (!std::same_as<T_To, typename T_End::value_type>)
            {
                return IdxRange{
                    pCast<T_To>(input.getBeginMd()),
                    pCast<T_To>(input.getEndMd()),
                    pCast<T_To>(input.getStrideMd())};
            }

            constexpr decltype(auto) operator()(auto&& input) const
                requires std::same_as<T_To, typename T_End::value_type>
            {
                return std::forward<decltype(input)>(input);
            }
        };

    } // namespace internal

    template<concepts::VectorOrScalar T_Extents>
    ALPAKA_FN_DG IdxRange(T_Extents const&) -> IdxRange<typename trait::GetVec_t<T_Extents>::UniVec>;

    template<concepts::VectorOrScalar T_Begin, concepts::VectorOrScalar T_End>
    ALPAKA_FN_DG IdxRange(T_Begin const&, T_End const&) -> IdxRange<
        typename trait::GetVec_t<T_Begin>::UniVec,
        typename trait::GetVec_t<T_End>::UniVec,
        typename trait::GetVec_t<T_End>::UniVec>;

    template<concepts::VectorOrScalar T_Begin, concepts::VectorOrScalar T_End, concepts::VectorOrScalar T_Stride>
    ALPAKA_FN_DG IdxRange(T_Begin const&, T_End const&, T_Stride const&) -> IdxRange<
        typename trait::GetVec_t<T_Begin>::UniVec,
        typename trait::GetVec_t<T_End>::UniVec,
        typename trait::GetVec_t<T_Stride>::UniVec>;

    namespace trait
    {
        template<typename T>
        struct IsIndexRange : std::false_type
        {
        };

        template<concepts::SpecializationOf<IdxRange> T>
        struct IsIndexRange<T> : std::true_type
        {
        };

        template<typename T>
        struct IsLazyIndexRange : std::false_type
        {
        };

    } // namespace trait

    template<typename T>
    constexpr bool isIndexRange_v = trait::IsIndexRange<T>::value;

    template<typename T>
    constexpr bool isLazyIndexRange_v = trait::IsLazyIndexRange<T>::value;

    namespace concepts
    {
        /** Concept to check if a type is an index range
         *
         * @tparam T Type to check
         * @tparam T_ValueType enforce a value type of the index range, if not provided the type is not checked
         * @tparam T_dim enforce a dimensionality of the index range, if not provided the value is not checked
         */
        template<typename T, typename T_ValueType = alpaka::NotRequired, uint32_t T_dim = alpaka::notRequiredDim>
        concept IdxRange
            = alpaka::isIndexRange_v<T>
              && (std::same_as<T_ValueType, typename T::IdxType> || std::same_as<T_ValueType, alpaka::NotRequired>)
              && ((T_dim == alpaka::notRequiredDim) || (T::dim() == T_dim));

        /** Concept to check if a type is a lazy-evaluated index range
         *
         * @attention the value type and dimension can not be evaluated for lazy index ranges.
         *
         * @tparam T Type to check
         */
        template<typename T>
        concept LazyIdxRange = alpaka::isLazyIndexRange_v<T>;

        template<typename T>
        concept IdxRangeDescription = alpaka::isIndexRange_v<T> || isLazyIndexRange_v<T>;

    } // namespace concepts
} // namespace alpaka
