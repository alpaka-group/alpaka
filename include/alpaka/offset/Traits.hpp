/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/core/Common.hpp>       // ALPAKA_FCT_HOST_ACC

#include <type_traits>                  // std::enable_if

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The offset specifics.
    //-----------------------------------------------------------------------------
    namespace offset
    {
        //-----------------------------------------------------------------------------
        //! The offset traits.
        //-----------------------------------------------------------------------------
        namespace traits
        {
            //#############################################################################
            //! The x offset get trait.
            //!
            //! If not specialized explicitly it returns 0.
            //#############################################################################
            template<
                typename TIdx,
                typename T,
                typename TSfinae = void>
            struct GetOffset
            {
                ALPAKA_FCT_HOST_ACC static auto getOffset(
                    T const &)
                -> Uint
                {
                    return 0u;
                }
            };

            //#############################################################################
            //! The x offset set trait.
            //#############################################################################
            template<
                typename TIdx,
                typename T,
                typename TSfinae = void>
            struct SetOffset;
        }

        //-----------------------------------------------------------------------------
        //! \return The offset in the given dimension.
        //-----------------------------------------------------------------------------
        template<
            Uint TuiIdx,
            typename TVal,
            typename TOffsets>
        ALPAKA_FCT_HOST_ACC auto getOffset(
            TOffsets const & offsets)
        -> TVal
        {
            return
                static_cast<TVal>(
                    traits::GetOffset<
                        std::integral_constant<Uint, TuiIdx>,
                        TOffsets>
                    ::getOffset(
                        offsets));
        }
        //-----------------------------------------------------------------------------
        //! \return The offset in x dimension.
        //-----------------------------------------------------------------------------
        template<
            typename TVal,
            typename TOffsets>
        ALPAKA_FCT_HOST_ACC auto getOffsetX(
            TOffsets const & offsets = TOffsets())
        -> decltype(getOffset<dim::DimT<TOffsets>::value - 1u, TVal>(offsets))
        {
            return getOffset<dim::DimT<TOffsets>::value - 1u, TVal>(offsets);
        }
        //-----------------------------------------------------------------------------
        //! \return The offset in y dimension.
        //-----------------------------------------------------------------------------
        template<
            typename TVal,
            typename TOffsets>
        ALPAKA_FCT_HOST_ACC auto getOffsetY(
            TOffsets const & offsets = TOffsets())
        -> decltype(getOffset<dim::DimT<TOffsets>::value - 2u, TVal>(offsets))
        {
            return getOffset<dim::DimT<TOffsets>::value - 2u, TVal>(offsets);
        }
        //-----------------------------------------------------------------------------
        //! \return The offset in z dimension.
        //-----------------------------------------------------------------------------
        template<
            typename TVal,
            typename TOffsets>
        ALPAKA_FCT_HOST_ACC auto getOffsetZ(
            TOffsets const & offsets = TOffsets())
        -> decltype(getOffset<dim::DimT<TOffsets>::value - 3u, TVal>(offsets))
        {
            return getOffset<dim::DimT<TOffsets>::value - 3u, TVal>(offsets);
        }

        //-----------------------------------------------------------------------------
        //! Sets the offset in the given dimension.
        //-----------------------------------------------------------------------------
        template<
            Uint TuiIdx,
            typename TOffsets,
            typename TVal>
        ALPAKA_FCT_HOST_ACC auto setOffset(
            TOffsets const & offsets,
            TVal const & offset)
        -> void
        {
            traits::SetOffset<
                std::integral_constant<Uint, TuiIdx>,
                TOffsets>
            ::setOffset(
                offsets,
                offset);
        }
        //-----------------------------------------------------------------------------
        //! Sets the offset in x dimension.
        //-----------------------------------------------------------------------------
        template<
            typename TOffsets,
            typename TVal>
        ALPAKA_FCT_HOST_ACC auto setOffsetX(
            TOffsets const & offsets,
            TVal const & offset)
        -> void
        {
            setOffset<dim::DimT<TOffsets>::value - 1u>(offsets, offset);
        }
        //-----------------------------------------------------------------------------
        //! Sets the offset in y dimension.
        //-----------------------------------------------------------------------------
        template<
            typename TOffsets,
            typename TVal>
        ALPAKA_FCT_HOST_ACC auto setOffsetY(
            TOffsets const & offsets,
            TVal const & offset)
        -> void
        {
            setOffset<dim::DimT<TOffsets>::value - 2u>(offsets, offset);
        }
        //-----------------------------------------------------------------------------
        //! Sets the offset in z dimension.
        //-----------------------------------------------------------------------------
        template<
            typename TOffsets,
            typename TVal>
        ALPAKA_FCT_HOST_ACC auto setOffsetZ(
            TOffsets const & offsets,
            TVal const & offset)
        -> void
        {
            setOffset<dim::DimT<TOffsets>::value - 3u>(offsets, offset);
        }

        //-----------------------------------------------------------------------------
        // Trait specializations for unsigned integral types.
        //-----------------------------------------------------------------------------
        namespace traits
        {
            //#############################################################################
            //! The unsigned integral x offset get trait specialization.
            //#############################################################################
            template<
                typename T>
            struct GetOffset<
                std::integral_constant<Uint, 0u>,
                T,
                typename std::enable_if<std::is_integral<T>::value && std::is_unsigned<T>::value>::type>
            {
                ALPAKA_FCT_HOST_ACC static auto getOffset(
                    T const & offset)
                -> Uint
                {
                    return static_cast<Uint>(offset);
                }
            };
        }
    }
}
