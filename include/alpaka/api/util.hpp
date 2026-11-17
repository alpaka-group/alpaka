/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/CVec.hpp"
#include "alpaka/api/trait.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/mem/DataPitches.hpp"
#include "alpaka/tag.hpp"
#include "alpaka/utility.hpp"

#include <cstdint>
#include <utility>

namespace alpaka::api::util
{
    namespace detail
    {
        template<
            std::integral auto T_limit,
            std::integral auto T_index,
            std::integral auto T_increment,
            size_t... T_idx>
        consteval auto adjustToLimit(concepts::CVector auto const input, std::index_sequence<T_idx...>)
        {
            if constexpr(input.product() <= static_cast<typename ALPAKA_TYPEOF(input)::value_type>(T_limit))
                return input;
            else
            {
                constexpr uint32_t dim = static_cast<uint32_t>(sizeof...(T_idx));

                constexpr auto newValue = CVec<
                    typename ALPAKA_TYPEOF(input)::value_type,
                    (T_idx == T_index
                         ? divExZero(input[T_idx], static_cast<typename ALPAKA_TYPEOF(input)::value_type>(2))
                         : input[T_idx])...>{};

                constexpr auto nextIncrement = dim == 1u ? 0u : T_increment;
                constexpr auto nextIdx = T_index + T_increment;

                if constexpr(nextIdx == dim)
                {
                    constexpr auto nextIncrement = dim == 1u ? 0u : -1u;

                    return adjustToLimit<T_limit, (dim == 1 ? 0 : dim - 1u), nextIncrement>(
                        newValue,
                        std::index_sequence<T_idx...>{});
                }
                else if constexpr(nextIdx == 0u)
                {
                    return adjustToLimit<T_limit, nextIdx, 1u>(newValue, std::index_sequence<T_idx...>{});
                }

                return adjustToLimit<T_limit, nextIdx, nextIncrement>(newValue, std::index_sequence<T_idx...>{});
            }
        }
    } // namespace detail

    /** adjust the input vector to a given limit by halving all components
     * until the product of these is below or equal to the limit */
    template<std::integral auto T_limit, std::integral auto T_index, std::integral auto T_increment>
    consteval auto adjustToLimit(concepts::CVector auto const input)
    {
        return detail::adjustToLimit<T_limit, T_index, T_increment>(
            input,
            std::make_index_sequence<ALPAKA_TYPEOF(input)::dim()>{});
    }

    /** adjust the input vector to a given limit by halving the largest dimension until the product of all components
     * is below or equal to the limit */
    inline auto adjustToLimit(concepts::Vector auto input, std::integral auto const limit)
    {
        using IdxType = typename ALPAKA_TYPEOF(input)::value_type;
        constexpr uint32_t dim = input.dim();
        IdxType limitValue = static_cast<IdxType>(limit);

        while(input.product() > limitValue)
        {
            uint32_t maxIdx = 0u;
            auto maxValue = input[0];
            for(auto i = 0u; i < dim; ++i)
                if(maxValue < input[i])
                {
                    maxIdx = i;
                    maxValue = input[i];
                }
            if(input.product() > limitValue)
                input[maxIdx] = divExZero(input[maxIdx], IdxType{2u});
        }
        return input;
    }

    /** provides a memory description to create multidimensional linewise aligned memory within a one dimensional
     * byte area
     *
     * @param alignmentInByte data alignment in bytes
     * @return tuple with the linearized data blob size in bytes and multi-dimensional pitches,
     * std::tuple(numBytes,pitchMD)
     */
    template<typename T_ValueType, alpaka::concepts::Vector T_Extents>
    inline auto emulatedAlignedMemDescription(uint32_t alignmentInByte, T_Extents extents)
    {
        constexpr auto dim = T_Extents::dim();
        if constexpr(dim == 1u)
        {
            size_t memSizeInByte = static_cast<size_t>(extents.x()) * sizeof(T_ValueType);
            alpaka::concepts::Vector auto pitches = typename T_Extents::UniVec{sizeof(T_ValueType)};
            return std::make_tuple(memSizeInByte, pitches);
        }
        else
        {
            using IdxType = typename T_Extents::value_type;
            auto alignment = static_cast<IdxType>(alignmentInByte);

            IdxType rowExtentInBytes = extents.x() * static_cast<IdxType>(sizeof(T_ValueType));
            IdxType rowPitchInBytes = alpaka::divCeil(rowExtentInBytes, alignment) * alignment;
            auto pitches = alpaka::calculatePitches<T_ValueType>(extents, rowPitchInBytes);

            size_t memSizeInByte = static_cast<size_t>(pitches[0]) * static_cast<size_t>(extents[0]);
            return std::make_tuple(memSizeInByte, pitches);
        }
    }

    consteval uint32_t highestPowerOfTwo(uint32_t value)
    {
        uint32_t result = 1u;
        while((result << 1u) <= value)
        {
            result <<= 1u;
        }
        return result;
    }

    /** Calculate the best alignment for SIMD optimized memory allocation
     *
     * @param api the API to use
     * @param deviceKind the device kind to use
     * @return the best alignment in bytes, will be a power of two value
     */
    template<typename T_ValueType>
    inline constexpr auto simdOptimizedAlignment(auto api, alpaka::concepts::DeviceKind auto deviceKind)
    {
        constexpr uint32_t typeAlignmentBytes = alignof(T_ValueType);
        constexpr uint32_t simdPackBytes
            = alpaka::getArchSimdWidth<T_ValueType>(api, deviceKind) * sizeof(T_ValueType);
        constexpr uint32_t bestSimdPackBytes = highestPowerOfTwo(simdPackBytes);
        constexpr uint32_t optimalAlignment = std::max(bestSimdPackBytes, typeAlignmentBytes);
        constexpr uint32_t adjustedAlignment = getAdjustedAlignment<T_ValueType>(api, deviceKind, optimalAlignment);
        return adjustedAlignment;
    }
} // namespace alpaka::api::util
