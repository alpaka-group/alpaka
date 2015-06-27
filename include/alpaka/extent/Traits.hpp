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
#include <alpaka/core/Fold.hpp>             // foldr

#if !defined(__CUDA_ARCH__)
    #include <boost/core/ignore_unused.hpp> // boost::ignore_unused
#endif

#include <type_traits>                      // std::enable_if
#include <functional>                       // std::multiplies

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The extent specifics.
    //-----------------------------------------------------------------------------
    namespace extent
    {
        //-----------------------------------------------------------------------------
        //! The extent traits.
        //-----------------------------------------------------------------------------
        namespace traits
        {
            //#############################################################################
            //! The extent get trait.
            //!
            //! If not specialized explicitly it returns 1.
            //#############################################################################
            template<
                typename TIdx,
                typename T,
                typename TSfinae = void>
            struct GetExtent
            {
                ALPAKA_FCT_HOST_ACC static auto getExtent(
                    T const &)
                -> Uint
                {
                    return static_cast<Uint>(1u);
                }
            };

            //#############################################################################
            //! The extent set trait.
            //#############################################################################
            template<
                typename TIdx,
                typename T,
                typename TSfinae = void>
            struct SetExtent;
        }

        //-----------------------------------------------------------------------------
        //! \return The width.
        //-----------------------------------------------------------------------------
        template<
            Uint TuiIdx,
            typename TVal,
            typename TExtents>
        ALPAKA_FCT_HOST_ACC auto getExtent(
            TExtents const & extents = TExtents())
        -> TVal
        {
            return
                static_cast<TVal>(
                    traits::GetExtent<
                        std::integral_constant<Uint, TuiIdx>,
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
        -> decltype(getExtent<dim::DimT<TExtents>::value - 1u, TVal>(extents))
        {
            return getExtent<dim::DimT<TExtents>::value - 1u, TVal>(extents);
        }
        //-----------------------------------------------------------------------------
        //! \return The height.
        //-----------------------------------------------------------------------------
        template<
            typename TVal,
            typename TExtents>
        ALPAKA_FCT_HOST_ACC auto getHeight(
            TExtents const & extents = TExtents())
        -> decltype(getExtent<dim::DimT<TExtents>::value - 2u, TVal>(extents))
        {
            return getExtent<dim::DimT<TExtents>::value - 2u, TVal>(extents);
        }
        //-----------------------------------------------------------------------------
        //! \return The depth.
        //-----------------------------------------------------------------------------
        template<
            typename TVal,
            typename TExtents>
        ALPAKA_FCT_HOST_ACC auto getDepth(
            TExtents const & extents = TExtents())
        -> decltype(getExtent<dim::DimT<TExtents>::value - 3u, TVal>(extents))
        {
            return getExtent<dim::DimT<TExtents>::value - 3u, TVal>(extents);
        }

        namespace detail
        {
            //-----------------------------------------------------------------------------
            //!
            //-----------------------------------------------------------------------------
            template<
                typename TVal,
                typename TExtents,
                size_t... TIndices>
            ALPAKA_FCT_HOST static auto getProductOfExtentsInternal(
                TExtents const & extents,
                alpaka::detail::index_sequence<TIndices...> const & indices)
            -> TVal
            {
#if !defined(__CUDA_ARCH__)
                boost::ignore_unused(indices);
#endif
                return
                    alpaka::foldr(
                        std::multiplies<TVal>(),
                        getExtent<TIndices, TVal>(extents)...);
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
            using IdxSequence = alpaka::detail::make_index_sequence<dim::DimT<TExtents>::value>;
            return detail::getProductOfExtentsInternal<TVal>(
                extents,
                IdxSequence());
        }

        //-----------------------------------------------------------------------------
        //! \return The width.
        //-----------------------------------------------------------------------------
        template<
            Uint TuiIdx,
            typename TExtents,
            typename TVal>
        ALPAKA_FCT_HOST_ACC auto setExtent(
            TExtents const & extents,
            TVal const & extent)
        -> void
        {
            return traits::SetExtent<
                std::integral_constant<Uint, TuiIdx>,
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
            setExtent<dim::DimT<TExtents>::value - 1u>(extents, width);
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
            setExtent<dim::DimT<TExtents>::value - 2u>(extents, height);
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
            setExtent<dim::DimT<TExtents>::value - 3u>(extents, depth);
        }

        //-----------------------------------------------------------------------------
        // Trait specializations for unsigned integral types.
        //-----------------------------------------------------------------------------
        namespace traits
        {
            //#############################################################################
            //! The unsigned integral width get trait specialization.
            //#############################################################################
            template<
                typename T>
            struct GetExtent<
                std::integral_constant<Uint, 0u>,
                T,
                typename std::enable_if<
                    std::is_integral<T>::value && std::is_unsigned<T>::value>::type>
            {
                ALPAKA_FCT_HOST_ACC static auto getExtent(
                    T const & extent)
                -> Uint
                {
                    return static_cast<Uint>(extent);
                }
            };
        }
    }
}
