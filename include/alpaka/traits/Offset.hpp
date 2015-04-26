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
    namespace traits
    {
        //-----------------------------------------------------------------------------
        //! The offset traits.
        //-----------------------------------------------------------------------------
        namespace offset
        {
            //#############################################################################
            //! The x offset get trait.
            //!
            //! If not specialized explicitly it returns 0.
            //#############################################################################
            template<
                UInt TuiIdx,
                typename T,
                typename TSfinae = void>
            struct GetOffset
            {
                ALPAKA_FCT_HOST_ACC static auto getOffset(
                    T const &)
                -> UInt
                {
                    return 0u;
                }
            };

            //#############################################################################
            //! The x offset set trait.
            //#############################################################################
            template<
                UInt TuiIdx,
                typename T,
                typename TSfinae = void>
            struct SetOffset;
        }
    }

    //-----------------------------------------------------------------------------
    //! The offset trait accessors.
    //-----------------------------------------------------------------------------
    namespace offset
    {
        //-----------------------------------------------------------------------------
        //! \return The offset in the given dimension.
        //-----------------------------------------------------------------------------
        template<
            UInt TuiIdx,
            typename TVal,
            typename TOffsets>
        ALPAKA_FCT_HOST_ACC auto getOffset(
            TOffsets const & offsets)
        -> TVal
        {
            return
                static_cast<TVal>(
                    traits::offset::GetOffset<
                        TuiIdx,
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
        -> decltype(getOffset<0u, TVal>(offsets))
        {
            return getOffset<0u, TVal>(offsets);
        }
        //-----------------------------------------------------------------------------
        //! \return The offset in y dimension.
        //-----------------------------------------------------------------------------
        template<
            typename TVal,
            typename TOffsets>
        ALPAKA_FCT_HOST_ACC auto getOffsetY(
            TOffsets const & offsets = TOffsets())
        -> decltype(getOffset<1u, TVal>(offsets))
        {
            return getOffset<1u, TVal>(offsets);
        }
        //-----------------------------------------------------------------------------
        //! \return The offset in z dimension.
        //-----------------------------------------------------------------------------
        template<
            typename TVal,
            typename TOffsets>
        ALPAKA_FCT_HOST_ACC auto getOffsetZ(
            TOffsets const & offsets = TOffsets())
        -> decltype(getOffset<2u, TVal>(offsets))
        {
            return getOffset<2u, TVal>(offsets);
        }

        //-----------------------------------------------------------------------------
        //! Sets the offset in the given dimension.
        //-----------------------------------------------------------------------------
        template<
            UInt TuiIdx,
            typename TOffsets,
            typename TVal>
        ALPAKA_FCT_HOST_ACC auto setOffset(
            TOffsets const & offsets,
            TVal const & offset)
        -> void
        {
            traits::offset::SetOffset<
                TuiIdx,
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
            setOffset<0u>(offsets, offset);
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
            setOffset<1u>(offsets, offset);
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
            setOffset<2u>(offsets, offset);
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for unsigned integral types.
    //-----------------------------------------------------------------------------
    namespace traits
    {
        namespace offset
        {
            //#############################################################################
            //! The unsigned integral x offset get trait specialization.
            //#############################################################################
            template<
                typename T>
            struct GetOffset<
                0u,
                T,
                typename std::enable_if<std::is_integral<T>::value && std::is_unsigned<T>::value>::type>
            {
                ALPAKA_FCT_HOST_ACC static auto getOffset(
                    T const & offset)
                -> UInt
                {
                    return static_cast<UInt>(offset);
                }
            };
        }
    }
}
