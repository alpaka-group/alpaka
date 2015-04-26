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

#include <alpaka/core/IntegerSequence.hpp>  // integer_sequence
#include <alpaka/core/Common.hpp>           // ALPAKA_FCT_HOST_ACC

#include <type_traits>                      // std::enable_if

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
            //! The extent get trait.
            //!
            //! If not specialized explicitly it returns 1.
            //#############################################################################
            template<
                UInt TuiIdx,
                typename T,
                typename TSfinae = void>
            struct GetExtent
            {
                ALPAKA_FCT_HOST_ACC static auto getExtent(
                    T const &)
                -> UInt
                {
                    return static_cast<UInt>(1u);
                }
            };

            //#############################################################################
            //! The extent set trait.
            //#############################################################################
            template<
                UInt TuiIdx,
                typename T,
                typename TSfinae = void>
            struct SetExtent;
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
            UInt TuiIdx,
            typename TVal,
            typename TExtents>
        ALPAKA_FCT_HOST_ACC auto getExtent(
            TExtents const & extents = TExtents())
        -> TVal
        {
            return
                static_cast<TVal>(
                    traits::extent::GetExtent<
                        TuiIdx,
                        TExtents>
                    ::getExtent(
                        extents));
        }
        //-----------------------------------------------------------------------------
        //! \return The width.
        //-----------------------------------------------------------------------------
        template<
            typename TVal,
            typename TExtents>
        ALPAKA_FCT_HOST_ACC auto getWidth(
            TExtents const & extents = TExtents())
        -> decltype(getExtent<0u, TVal>(extents))
        {
            return getExtent<0u, TVal>(extents);
        }
        //-----------------------------------------------------------------------------
        //! \return The height.
        //-----------------------------------------------------------------------------
        template<
            typename TVal,
            typename TExtents>
        ALPAKA_FCT_HOST_ACC auto getHeight(
            TExtents const & extents = TExtents())
        -> decltype(getExtent<1u, TVal>(extents))
        {
            return getExtent<1u, TVal>(extents);
        }
        //-----------------------------------------------------------------------------
        //! \return The depth.
        //-----------------------------------------------------------------------------
        template<
            typename TVal,
            typename TExtents>
        ALPAKA_FCT_HOST_ACC auto getDepth(
            TExtents const & extents = TExtents())
        -> decltype(getExtent<2u, TVal>(extents))
        {
            return getExtent<2u, TVal>(extents);
        }

        namespace detail
        {
            //-----------------------------------------------------------------------------
            //!
            //-----------------------------------------------------------------------------
            template<
                typename T>
            T multiply(
                T const & t)
            {
                return t;
            }
            //-----------------------------------------------------------------------------
            //!
            //-----------------------------------------------------------------------------
            template<
                typename T0,
                typename T1,
                typename... Ts>
            auto multiply(
                T0 const & t0, 
                T1 const & t1, 
                Ts const & ... ts)
            -> decltype(t0 * multiply(t1, ts...))
            {
                return t0 * multiply(t1, ts...);
            }

            //-----------------------------------------------------------------------------
            //!
            //-----------------------------------------------------------------------------
            template<
                typename TVal,
                typename TExtents,
                size_t... TIndices>
            ALPAKA_FCT_HOST static auto getProductOfExtentsInternal(
                TExtents const & extents,
#if !BOOST_COMP_MSVC     // MSVC 190022512 introduced a new bug with alias templates: error C3520: 'TIndices': parameter pack must be expanded in this context
            alpaka::detail::index_sequence<TIndices...> const &)
#else
            alpaka::detail::integer_sequence<std::size_t, TIndices...> const &)
#endif
            -> TVal
            {
                return multiply(getExtent<TIndices, TVal>(extents)...);
            } 
        }

        //-----------------------------------------------------------------------------
        //! \return The product of the extents.
        //-----------------------------------------------------------------------------
        template<
            typename TVal,
            typename TExtents>
        ALPAKA_FCT_HOST_ACC auto getProductOfExtents(
            TExtents const & extents = TExtents())
        -> TVal
        {
#if (BOOST_COMP_MSVC) && (BOOST_COMP_MSVC < BOOST_VERSION_NUMBER(14, 0, 0))
            using IdxSequence = typename alpaka::detail::make_index_sequence<dim::DimT<TExtents>::value>::type;
#else
            using IdxSequence = alpaka::detail::make_index_sequence<dim::DimT<TExtents>::value>;
#endif
            return detail::getProductOfExtentsInternal<TVal>(
                extents,
                IdxSequence());
        }
        
        //-----------------------------------------------------------------------------
        //! \return The width.
        //-----------------------------------------------------------------------------
        template<
            UInt TuiIdx,
            typename TExtents,
            typename TVal>
        ALPAKA_FCT_HOST_ACC auto setExtent(
            TExtents const & extents,
            TVal const & extent)
        -> void
        {
            return traits::extent::SetExtent<
                TuiIdx,
                TExtents>
            ::setExtent(
                extents,
                extent);
        }
        //-----------------------------------------------------------------------------
        //! Sets the width.
        //-----------------------------------------------------------------------------
        template<
            typename TExtents,
            typename TVal>
        ALPAKA_FCT_HOST_ACC auto setWidth(
            TExtents const & extents,
            TVal const & width)
        -> void
        {
            setExtent<0u>(extents, width);
        }
        //-----------------------------------------------------------------------------
        //! Sets the height.
        //-----------------------------------------------------------------------------
        template<
            typename TExtents,
            typename TVal>
        ALPAKA_FCT_HOST_ACC auto setHeight(
            TExtents const & extents,
            TVal const & height)
        -> void
        {
            setExtent<1u>(extents, height);
        }
        //-----------------------------------------------------------------------------
        //! Sets the depth.
        //-----------------------------------------------------------------------------
        template<
            typename TExtents,
            typename TVal>
        ALPAKA_FCT_HOST_ACC auto setDepth(
            TExtents const & extents,
            TVal const & depth)
        -> void
        {
            setExtent<2u>(extents, depth);
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
            struct GetExtent<
                0u,
                T,
                typename std::enable_if<
                    std::is_integral<T>::value && std::is_unsigned<T>::value>::type>
            {
                ALPAKA_FCT_HOST_ACC static auto getExtent(
                    T const & extent)
                -> UInt
                {
                    return static_cast<UInt>(extent);
                }
            };
        }
    }
}
