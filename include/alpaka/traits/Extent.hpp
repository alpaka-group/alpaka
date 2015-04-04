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
        //! The extent traits.
        //-----------------------------------------------------------------------------
        namespace extent
        {
            //#############################################################################
            //! The extents get trait.
            //#############################################################################
            template<
                typename T,
                typename TSfinae = void>
            struct GetExtents;

            //#############################################################################
            //! The width get trait.
            //!
            //! If not specialized explicitly it returns 1.
            //#############################################################################
            template<
                typename T,
                typename TSfinae = void>
            struct GetWidth
            {
                ALPAKA_FCT_HOST_ACC static auto getWidth(
                    T const &)
                -> UInt
                {
                    return static_cast<UInt>(1u);
                }
            };
            //#############################################################################
            //! The height get trait.
            //!
            //! If not specialized explicitly it returns 1.
            //#############################################################################
            template<
                typename T,
                typename TSfinae = void>
            struct GetHeight
            {
                ALPAKA_FCT_HOST_ACC static auto getHeight(
                    T const &)
                -> UInt
                {
                    return static_cast<UInt>(1u);
                }
            };
            //#############################################################################
            //! The depth get trait.
            //!
            //! If not specialized explicitly it returns 1.
            //#############################################################################
            template<
                typename T,
                typename TSfinae = void>
            struct GetDepth
            {
                ALPAKA_FCT_HOST_ACC static auto getDepth(
                    T const &)
                -> UInt
                {
                    return static_cast<UInt>(1u);
                }
            };

            //#############################################################################
            //! The width set trait.
            //#############################################################################
            template<
                typename T,
                typename TSfinae = void>
            struct SetWidth;
            //#############################################################################
            //! The height set trait.
            //#############################################################################
            template<
                typename T,
                typename TSfinae = void>
            struct SetHeight;
            //#############################################################################
            //! The depth set trait.
            //#############################################################################
            template<
                typename T,
                typename TSfinae = void>
            struct SetDepth;
        }
    }

    //-----------------------------------------------------------------------------
    //! The extent trait accessors.
    //-----------------------------------------------------------------------------
    namespace extent
    {
        //-----------------------------------------------------------------------------
        //! \return The extents.
        //-----------------------------------------------------------------------------
        template<
            typename TExtents>
        ALPAKA_FCT_HOST_ACC auto getExtents(
            TExtents const & extents = TExtents())
        -> decltype(traits::extent::GetExtents<TExtents>::getExtents(std::declval<TExtents const &>()))
        {
            return traits::extent::GetExtents<
                TExtents>
            ::template getExtents(
                extents);
        }
        //-----------------------------------------------------------------------------
        //! \return The width.
        //-----------------------------------------------------------------------------
        template<
            typename TExtents>
        ALPAKA_FCT_HOST_ACC auto getWidth(
            TExtents const & extents = TExtents())
        -> UInt
        {
            return traits::extent::GetWidth<
                TExtents>
            ::getWidth(
                extents);
        }
        //-----------------------------------------------------------------------------
        //! \return The height.
        //-----------------------------------------------------------------------------
        template<
            typename TExtents>
        ALPAKA_FCT_HOST_ACC auto getHeight(
            TExtents const & extents = TExtents())
        -> UInt
        {
            return traits::extent::GetHeight<
                TExtents>
            ::getHeight(
                extents);
        }
        //-----------------------------------------------------------------------------
        //! \return The depth.
        //-----------------------------------------------------------------------------
        template<
            typename TExtents>
        ALPAKA_FCT_HOST_ACC auto getDepth(
            TExtents const & extents = TExtents())
        -> UInt
        {
            return traits::extent::GetDepth<
                TExtents>
            ::getDepth(
                extents);
        }
        //-----------------------------------------------------------------------------
        //! \return The product of the extents.
        //-----------------------------------------------------------------------------
        template<
            typename TExtents>
        ALPAKA_FCT_HOST_ACC auto getProductOfExtents(
            TExtents const & extents = TExtents())
        -> UInt
        {
            return
                getWidth(extents)
                * getHeight(extents)
                * getDepth(extents);
        }

        //-----------------------------------------------------------------------------
        //! Sets the width.
        //-----------------------------------------------------------------------------
        template<
            typename TExtents,
            typename TInt>
        ALPAKA_FCT_HOST_ACC auto setWidth(
            TExtents const & extents,
            TInt const & width)
        -> void
        {
            traits::extent::SetWidth<
                TExtents>
            ::setWidth(
                extents,
                width);
        }
        //-----------------------------------------------------------------------------
        //! Sets the height.
        //-----------------------------------------------------------------------------
        template<
            typename TExtents,
            typename TInt>
        ALPAKA_FCT_HOST_ACC auto setHeight(
            TExtents const & extents,
            TInt const & height)
        -> void
        {
            traits::extent::SetHeight<
                TExtents>
            ::setHeight(
                extents,
                height);
        }
        //-----------------------------------------------------------------------------
        //! Sets the depth.
        //-----------------------------------------------------------------------------
        template<
            typename TExtents,
            typename TInt>
        ALPAKA_FCT_HOST_ACC auto setDepth(
            TExtents const & extents,
            TInt const & depth)
        -> void
        {
            traits::extent::SetDepth<
                TExtents>
            ::setDepth(
                extents,
                depth);
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for unsigned integral types.
    //-----------------------------------------------------------------------------
    namespace traits
    {
        namespace extent
        {
            //#############################################################################
            //! The unsigned integral width get trait specialization.
            //#############################################################################
            template<
                typename T>
            struct GetWidth<
                T,
                typename std::enable_if<std::is_integral<T>::value && std::is_unsigned<T>::value>::type>
            {
                ALPAKA_FCT_HOST_ACC static auto getWidth(
                    T const & extent)
                -> UInt
                {
                    return static_cast<UInt>(extent);
                }
            };
        }
    }
}
