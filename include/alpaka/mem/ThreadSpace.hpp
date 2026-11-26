/* Copyright 2024 Andrea Bocci, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/CVec.hpp"
#include "alpaka/Vec.hpp"
#include "alpaka/core/common.hpp"

#include <cstdint>

namespace alpaka
{
    template<concepts::Vector T_ThreadIdx, concepts::Vector T_ThreadCount>
    struct ThreadSpace
    {
        constexpr ThreadSpace(T_ThreadIdx const& threadIdx, T_ThreadCount const& threadCount)
            : m_threadIdx(threadIdx)
            , m_threadCount(threadCount)
        {
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
            stream << m_threadIdx << separator << m_threadCount;
            stream << localeEnclosingEnd;
            return stream.str();
        }

        template<concepts::CVector T_CSelect>
        constexpr ThreadSpace mapTo(T_CSelect selection) const requires(T_ThreadIdx::dim() <= T_CSelect::dim())
        {
            alpaka::unused(selection);
            static_assert(T_ThreadIdx::dim() == T_CSelect::dim(), "can not map to a larger dimension");
            return *this;
        }

        template<concepts::CVector T_CSelect>
        constexpr auto mapTo(T_CSelect selection) const requires(T_ThreadIdx::dim() > T_CSelect::dim())
        {
            alpaka::unused(selection);

            using IdxType = typename T_ThreadIdx::value_type;
            constexpr uint32_t dim = T_ThreadIdx::dim();

            auto allElements = iotaCVec<IdxType, dim>();
            constexpr auto notSelectedDims = filter(allElements, T_CSelect{});

            // Transform into a universal vector because the input could be a CVec which is read only.
            auto threadIndex = typename ALPAKA_TYPEOF(m_threadIdx)::UniVec{m_threadIdx};
            auto numThreads = typename ALPAKA_TYPEOF(m_threadCount)::UniVec{m_threadCount};

            // map not selected dimensions to the slowest selected dimension
            for(uint32_t x = 0u; x < notSelectedDims.dim(); ++x)
            {
                auto d = notSelectedDims[x];
                auto old = threadIndex[d];
                threadIndex[d] = 0u;
                threadIndex[T_CSelect{}[0]] += old * numThreads[T_CSelect{}[0]];
            }

            for(uint32_t x = 0u; x < notSelectedDims.dim(); ++x)
            {
                auto d = notSelectedDims[x];
                auto old = numThreads[d];
                numThreads[d] = 1u;
                numThreads[T_CSelect{}[0]] *= old;
            }

            return ThreadSpace<ALPAKA_TYPEOF(threadIndex), ALPAKA_TYPEOF(numThreads)>{threadIndex, numThreads};
        }

        [[nodiscard]] constexpr auto getThreadIdx() const -> T_ThreadIdx
        {
            return m_threadIdx;
        }

        [[nodiscard]] constexpr auto getThreadCount() const -> T_ThreadCount
        {
            return m_threadCount;
        }

        using value_type = typename T_ThreadIdx::value_type;

    private:
        T_ThreadIdx m_threadIdx;
        T_ThreadCount m_threadCount;
    };

    namespace internal
    {
        template<typename T_To, typename T_ThreadIdx, typename T_ThreadCount>
        struct PCast::Op<T_To, ThreadSpace<T_ThreadIdx, T_ThreadCount>>
        {
            constexpr auto operator()(auto&& input) const
                requires std::convertible_to<typename T_ThreadIdx::value_type, T_To>
                         && (!std::same_as<T_To, typename T_ThreadIdx::value_type>)
            {
                return ThreadSpace{pCast<T_To>(input.getThreadIdx()), pCast<T_To>(input.getThreadCount())};
            }

            constexpr decltype(auto) operator()(auto&& input) const
                requires std::same_as<T_To, typename T_ThreadIdx::value_type>
            {
                return std::forward<decltype(input)>(input);
            }
        };
    } // namespace internal

    template<std::size_t I, typename T_ThreadIdx, typename T_ThreadCount>
    constexpr auto get(alpaka::ThreadSpace<T_ThreadIdx, T_ThreadCount> const& v) -> T_ThreadIdx requires(I == 0u)
    {
        return v.getThreadIdx();
    }

    template<std::size_t I, typename T_ThreadIdx, typename T_ThreadCount>
    constexpr auto get(alpaka::ThreadSpace<T_ThreadIdx, T_ThreadCount>& v) -> T_ThreadIdx const& requires(I == 0u)
    {
        return v.getThreadIdx();
    }

    template<std::size_t I, typename T_ThreadIdx, typename T_ThreadCount>
    constexpr auto get(alpaka::ThreadSpace<T_ThreadIdx, T_ThreadCount> const& v) -> T_ThreadCount requires(I == 1u)
    {
        return v.getThreadCount();
    }

    template<std::size_t I, typename T_ThreadIdx, typename T_ThreadCount>
    constexpr auto get(alpaka::ThreadSpace<T_ThreadIdx, T_ThreadCount>& v) -> T_ThreadCount const& requires(I == 1u)
    {
        return v.getThreadCount();
    }

} // namespace alpaka

namespace std
{
    template<typename T_ThreadIdx, typename T_ThreadCount>
    struct tuple_size<alpaka::ThreadSpace<T_ThreadIdx, T_ThreadCount>>
    {
        static constexpr std::size_t value = 2u;
    };

    template<std::size_t I, typename T_ThreadIdx, typename T_ThreadCount>
    struct tuple_element<I, alpaka::ThreadSpace<T_ThreadIdx, T_ThreadCount>>
    {
        using type = std::conditional_t<I == 0u, T_ThreadIdx, T_ThreadCount>;
    };
} // namespace std
