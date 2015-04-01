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
            //! The offsets get trait.
            //#############################################################################
            template<
                typename T, 
                typename TSfinae = void>
            struct GetOffsets;

            //#############################################################################
            //! The x offset get trait.
            //!
            //! If not specialized explicitly it returns 0.
            //#############################################################################
            template<
                typename T, 
                typename TSfinae = void>
            struct GetOffsetX
            {
                ALPAKA_FCT_HOST_ACC static auto getOffsetX(
                    T const &)
                -> UInt
                {
                    return static_cast<UInt>(0u);
                }
            };
            //#############################################################################
            //! The y offset get trait.
            //!
            //! If not specialized explicitly it returns 0.
            //#############################################################################
            template<
                typename T, 
                typename TSfinae = void>
            struct GetOffsetY
            {
                ALPAKA_FCT_HOST_ACC static auto getOffsetY(
                    T const &)
                -> UInt
                {
                    return static_cast<UInt>(0u);
                }
            };
            //#############################################################################
            //! The z offset get trait.
            //!
            //! If not specialized explicitly it returns 0.
            //#############################################################################
            template<
                typename T, 
                typename TSfinae = void>
            struct GetOffsetZ
            {
                ALPAKA_FCT_HOST_ACC static auto getOffsetZ(
                    T const &)
                -> UInt
                {
                    return static_cast<UInt>(0u);
                }
            };

            //#############################################################################
            //! The x offset set trait.
            //#############################################################################
            template<
                typename T, 
                typename TSfinae = void>
            struct SetOffsetX;
            //#############################################################################
            //! The y offset set trait.
            //#############################################################################
            template<
                typename T, 
                typename TSfinae = void>
            struct SetOffsetY;
            //#############################################################################
            //! The z offset set trait.
            //#############################################################################
            template<
                typename T, 
                typename TSfinae = void>
            struct SetOffsetZ;
        }
    }

    //-----------------------------------------------------------------------------
    //! The offset trait accessors.
    //-----------------------------------------------------------------------------
    namespace offset
    {
        //-----------------------------------------------------------------------------
        //! \return The offsets.
        //-----------------------------------------------------------------------------
        template<
            typename TOffsets>
        ALPAKA_FCT_HOST_ACC auto getOffsets(
            TOffsets const & offsets)
        -> decltype(traits::offset::GetOffsets<TOffsets>::getOffsets(std::declval<TOffsets const &>()))
        {
            return traits::offset::GetOffsets<
                TOffsets>
            ::getOffsets(
                offsets);
        }
        //-----------------------------------------------------------------------------
        //! \return The x offset.
        //-----------------------------------------------------------------------------
        template<
            typename TOffsets>
        ALPAKA_FCT_HOST_ACC auto getOffsetX(
            TOffsets const & offsets)
        -> UInt
        {
            return traits::offset::GetOffsetX<
                TOffsets>
            ::getOffsetX(
                offsets);
        }
        //-----------------------------------------------------------------------------
        //! \return The y offset.
        //-----------------------------------------------------------------------------
        template<
            typename TOffsets>
        ALPAKA_FCT_HOST_ACC auto getOffsetY(
            TOffsets const & offsets)
        -> UInt
        {
            return traits::offset::GetOffsetY<
                TOffsets>
            ::getOffsetY(
                offsets);
        }
        //-----------------------------------------------------------------------------
        //! \return The z offset.
        //-----------------------------------------------------------------------------
        template<
            typename TOffsets>
        ALPAKA_FCT_HOST_ACC auto getOffsetZ(
            TOffsets const & offsets)
        -> UInt
        {
            return traits::offset::GetOffsetZ<
                TOffsets>
            ::getOffsetZ(
                offsets);
        }

        //-----------------------------------------------------------------------------
        //! Sets the x offset.
        //-----------------------------------------------------------------------------
        template<
            typename TOffsets,
            typename TInt>
        ALPAKA_FCT_HOST_ACC auto setOffsetX(
            TOffsets const & offsets,
            TInt const & xOffset)
        -> void
        {
            traits::offset::SetOffsetX<
                TOffsets>
            ::setOffsetX(
                offsets, 
                xOffset);
        }
        //-----------------------------------------------------------------------------
        //! Sets the y offset.
        //-----------------------------------------------------------------------------
        template<
            typename TOffsets,
            typename TInt>
        ALPAKA_FCT_HOST_ACC auto setOffsetY(
            TOffsets const & offsets,
            TInt const & yOffset)
        -> void
        {
            traits::offset::SetOffsetY<
                TOffsets>
            ::setOffsetY(
                offsets, 
                yOffset);
        }
        //-----------------------------------------------------------------------------
        //! Sets the z offset.
        //-----------------------------------------------------------------------------
        template<
            typename TOffsets,
            typename TInt>
        ALPAKA_FCT_HOST_ACC auto setOffsetZ(
            TOffsets const & offsets,
            TInt const & zOffset)
        -> void
        {
            traits::offset::SetOffsetZ<
                TOffsets>
            ::setOffsetZ(
                offsets, 
                zOffset);
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
            struct GetOffsetX<
                T,
                typename std::enable_if<std::is_integral<T>::value && std::is_unsigned<T>::value>::type>
            {
                ALPAKA_FCT_HOST_ACC static auto getOffsetX(
                    T const & offset)
                -> UInt
                {
                    return static_cast<UInt>(offset);
                }
            };
        }
    }
}
