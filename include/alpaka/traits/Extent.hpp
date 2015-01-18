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

#include <alpaka/core/Common.hpp>   // ALPAKA_FCT_HOST_ACC

#include <cstdint>                  // std::size_t

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
            //! The width get trait.
            //!
            //! If not specialized explicitly it returns 1.
            //#############################################################################
            template<
                typename T, 
                typename TSfinae = void>
            struct GetWidth
            {
                ALPAKA_FCT_HOST_ACC static std::size_t getWidth(
                    T const &)
                {
                    return 1;
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
                ALPAKA_FCT_HOST_ACC static std::size_t getHeight(
                    T const &)
                {
                    return 1;
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
                ALPAKA_FCT_HOST_ACC static std::size_t getDepth(
                    T const &)
                {
                    return 1;
                }
            };
        }
    }

    //-----------------------------------------------------------------------------
    //! The extent trait accessors.
    //-----------------------------------------------------------------------------
    namespace extent
    {
        //-----------------------------------------------------------------------------
        //! \return The width.
        //-----------------------------------------------------------------------------
        template<
            typename T>
        ALPAKA_FCT_HOST_ACC std::size_t getWidth(
            T const & width)
        {
            return traits::extent::GetWidth<T>::getWidth(width);
        }
        //-----------------------------------------------------------------------------
        //! \return The height.
        //-----------------------------------------------------------------------------
        template<
            typename T>
        ALPAKA_FCT_HOST_ACC std::size_t getHeight(
            T const & height)
        {
            return traits::extent::GetHeight<T>::getHeight(height);
        }
        //-----------------------------------------------------------------------------
        //! \return The depth.
        //-----------------------------------------------------------------------------
        template<
            typename T>
        ALPAKA_FCT_HOST_ACC std::size_t getDepth(
            T const & depth)
        {
            return traits::extent::GetDepth<T>::getDepth(depth);
        }
        //-----------------------------------------------------------------------------
        //! \return The product of the extents.
        //-----------------------------------------------------------------------------
        template<
            typename T>
        ALPAKA_FCT_HOST_ACC std::size_t getProductOfExtents(
            T const & extents)
        {
            return
                getWidth(extents)
                * getHeight(extents)
                * getDepth(extents);
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for unsigned integral types.
    //-----------------------------------------------------------------------------
    namespace traits
    {
        namespace dim
        {
            //#############################################################################
            //! The  unsigned integral dimension getter trait specialization.
            //#############################################################################
            template<
                typename T>
            struct GetDim<
                T,
                typename std::enable_if<std::is_integral<T>::value && std::is_unsigned<T>::value, void>::type
            >
            {
                using type = alpaka::dim::Dim<std::rank<T>::value>;
            };
        }

        namespace extent
        {
            //#############################################################################
            //! The unsigned integral width get trait specialization.
            //#############################################################################
            template<
                typename T>
            struct GetWidth<
                T,
                typename std::enable_if<std::is_integral<T>::value && std::is_unsigned<T>::value, void>::type
            >
            {
                static std::size_t getWidth(
                    T const & extent)
                {
                    return static_cast<std::size_t>(extent);
                }
            };
        }
    }
}