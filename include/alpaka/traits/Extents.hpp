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
        //! The extents traits.
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
                    return 1u;
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
                    return 1u;
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
                    return 1u;
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
    //! The extents trait accessors.
    //-----------------------------------------------------------------------------
    namespace extent
    {
        //-----------------------------------------------------------------------------
        //! \return The width.
        //-----------------------------------------------------------------------------
        template<
            typename TExtents>
        ALPAKA_FCT_HOST_ACC std::size_t getWidth(
            TExtents const & extents)
        {
            return traits::extent::GetWidth<TExtents>::getWidth(extents);
        }
        //-----------------------------------------------------------------------------
        //! \return The height.
        //-----------------------------------------------------------------------------
        template<
            typename TExtents>
        ALPAKA_FCT_HOST_ACC std::size_t getHeight(
            TExtents const & extents)
        {
            return traits::extent::GetHeight<TExtents>::getHeight(extents);
        }
        //-----------------------------------------------------------------------------
        //! \return The depth.
        //-----------------------------------------------------------------------------
        template<
            typename TExtents>
        ALPAKA_FCT_HOST_ACC std::size_t getDepth(
            TExtents const & extents)
        {
            return traits::extent::GetDepth<TExtents>::getDepth(extents);
        }
        //-----------------------------------------------------------------------------
        //! \return The product of the extents.
        //-----------------------------------------------------------------------------
        template<
            typename TExtents>
        ALPAKA_FCT_HOST_ACC std::size_t getProductOfExtents(
            TExtents const & extents)
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
            typename TExtents>
            ALPAKA_FCT_HOST_ACC void setWidth(
            TExtents const & extents,
            std::size_t const & width)
        {
            return traits::extent::SetWidth<TExtents>::setWidth(extents, width);
        }
        //-----------------------------------------------------------------------------
        //! Sets the height.
        //-----------------------------------------------------------------------------
        template<
            typename TExtents>
        ALPAKA_FCT_HOST_ACC std::size_t setHeight(
            TExtents const & extents,
            std::size_t const & height)
        {
            return traits::extent::SetHeight<TExtents>::setHeight(extents, height);
        }
        //-----------------------------------------------------------------------------
        //! Sets the depth.
        //-----------------------------------------------------------------------------
        template<
            typename TExtents>
        ALPAKA_FCT_HOST_ACC std::size_t setDepth(
            TExtents const & extents,
            std::size_t const & depth)
        {
            return traits::extent::SetDepth<TExtents>::setDepth(extents, depth);
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
            //! The unsigned integral dimension getter trait specialization.
            //#############################################################################
            template<
                typename T>
            struct GetDim<
                T,
                typename std::enable_if<std::is_integral<T>::value && std::is_unsigned<T>::value>::type>
            {
                using type = alpaka::dim::Dim1;
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
                typename std::enable_if<std::is_integral<T>::value && std::is_unsigned<T>::value>::type>
            {
                ALPAKA_FCT_HOST_ACC static std::size_t getWidth(
                    T const & extent)
                {
                    return static_cast<std::size_t>(extent);
                }
            };
        }
    }
}