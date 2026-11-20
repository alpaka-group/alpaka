/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/CVec.hpp"
#include "alpaka/Vec.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/mem/Alignment.hpp"

#include <type_traits>

namespace alpaka
{
    //! Calculate the pitches purely from the extents.
    template<typename T_Elem, alpaka::concepts::Vector T_Vec>
    constexpr auto calculatePitchesFromExtents(T_Vec const& extent)
    {
        constexpr auto dim = T_Vec::dim();
        using Type = typename T_Vec::value_type;
        typename T_Vec::UniVec pitchBytes;
        if constexpr(dim > 0)
            pitchBytes.back() = static_cast<Type>(sizeof(T_Elem));
        if constexpr(dim > 1)
            for(Type i = dim - 1; i > 0; i--)
                pitchBytes[i - 1] = extent[i] * pitchBytes[i];
        return pitchBytes;
    }

    //! Calculate the pitches purely from the extents.
    template<typename T_Elem, alpaka::concepts::Vector T_Vec>
    requires(T_Vec::dim() >= 2)
    constexpr auto calculatePitches(T_Vec const& extent, typename T_Vec::value_type const& rowPitchBytes)
    {
        constexpr auto dim = T_Vec::dim();
        using Type = typename T_Vec::value_type;
        typename T_Vec::UniVec pitchBytes;
        pitchBytes.back() = static_cast<Type>(sizeof(T_Elem));
        if constexpr(dim > 1)
            pitchBytes[dim - 2u] = rowPitchBytes;
        if constexpr(dim > 2)
            for(Type i = dim - 2; i > 0; i--)
                pitchBytes[i - 1] = extent[i] * pitchBytes[i];
        return pitchBytes;
    }

    template<typename T_Type, concepts::Vector T_Pitches>
    struct DataPitches
    {
        using value_type = T_Type;
        using index_type = typename T_Pitches::value_type;

        static consteval uint32_t dim()
        {
            return T_Pitches::dim();
        }

        constexpr DataPitches(T_Pitches const& pitchBytes) : m_pitch(pitchBytes.eraseBack())
        {
            assert(pitchBytes.back() == sizeof(value_type));
        }

        /*Object must init by copy a valid instance*/
        constexpr DataPitches() = default;

        constexpr auto getPitches() const
        {
            Vec<index_type, dim()> result;
            for(uint32_t d = 0u; d < dim() - 1u; ++d)
            {
                result[d] = m_pitch[d];
            }
            result.back() = static_cast<index_type>(sizeof(value_type));
            return result;
        }

        constexpr index_type operator[](std::integral auto idx) const
        {
            return getPitches()[idx];
        }

    private:
        decltype(std::declval<T_Pitches>().eraseBack()) m_pitch;
    };

    template<typename T_Type, typename T_IndexType, typename T_Storage>
    struct DataPitches<T_Type, Vec<T_IndexType, 1u, T_Storage>>
    {
        using value_type = T_Type;
        using index_type = T_IndexType;

        static consteval uint32_t dim()
        {
            return 1u;
        }

        constexpr DataPitches([[maybe_unused]] Vec<T_IndexType, 1u> const& pitchBytes)
        {
            assert(pitchBytes.back() == sizeof(value_type));
        }

        /*Object must init by copy a valid instance*/
        constexpr DataPitches() = default;

        constexpr auto getPitches() const
        {
            return Vec{static_cast<index_type>(sizeof(value_type))};
        }
    };
} // namespace alpaka
