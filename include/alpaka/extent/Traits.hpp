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
#include <alpaka/core/Common.hpp>           // ALPAKA_FN_HOST_ACC
#include <alpaka/core/Fold.hpp>             // core::foldr

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
                typename TExtents,
                typename TSfinae = void>
            struct GetExtent
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getExtent(
                    TExtents const &)
                -> size::Size<TExtents>
                {
                    return static_cast<size::Size<TExtents>>(1);
                }
            };

            //#############################################################################
            //! The extent set trait.
            //#############################################################################
            template<
                typename TIdx,
                typename TExtents,
                typename TExtent,
                typename TSfinae = void>
            struct SetExtent;
        }

        //-----------------------------------------------------------------------------
        //! \return The extent in the given dimension.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            std::size_t Tidx,
            typename TExtents>
        ALPAKA_FN_HOST_ACC auto getExtent(
            TExtents const & extents = TExtents())
        -> size::Size<TExtents>
        {
            return
                traits::GetExtent<
                    std::integral_constant<std::size_t, Tidx>,
                    TExtents>
                ::getExtent(
                    extents);
        }
        //-----------------------------------------------------------------------------
        //! \return The width.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TExtents>
        ALPAKA_FN_HOST_ACC auto getWidth(
            TExtents const & extents = TExtents())
        -> size::Size<TExtents>
        {
            return getExtent<dim::Dim<TExtents>::value - 1u>(extents);
        }
        //-----------------------------------------------------------------------------
        //! \return The height.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TExtents>
        ALPAKA_FN_HOST_ACC auto getHeight(
            TExtents const & extents = TExtents())
        -> size::Size<TExtents>
        {
            return getExtent<dim::Dim<TExtents>::value - 2u>(extents);
        }
        //-----------------------------------------------------------------------------
        //! \return The depth.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TExtents>
        ALPAKA_FN_HOST_ACC auto getDepth(
            TExtents const & extents = TExtents())
        -> size::Size<TExtents>
        {
            return getExtent<dim::Dim<TExtents>::value - 3u>(extents);
        }

        namespace detail
        {
            //-----------------------------------------------------------------------------
            //!
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename TExtents,
                size_t... TIndices>
            ALPAKA_FN_HOST static auto getProductOfExtentsInternal(
                TExtents const & extents,
                alpaka::core::detail::index_sequence<TIndices...> const & indices)
            -> size::Size<TExtents>
            {
#if !defined(__CUDA_ARCH__)
                boost::ignore_unused(indices);
#endif
                return
                    core::foldr(
                        std::multiplies<size::Size<TExtents>>(),
                        getExtent<TIndices>(extents)...);
            }
        }

        //-----------------------------------------------------------------------------
        //! \return The product of the extents.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TExtents>
        ALPAKA_FN_HOST_ACC auto getProductOfExtents(
            TExtents const & extents = TExtents())
        -> size::Size<TExtents>
        {
            using IdxSequence = alpaka::core::detail::make_index_sequence<dim::Dim<TExtents>::value>;
            return
                detail::getProductOfExtentsInternal(
                    extents,
                    IdxSequence());
        }

        //-----------------------------------------------------------------------------
        //! Sets the extent in the given dimension.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            std::size_t Tidx,
            typename TExtents,
            typename TExtent>
        ALPAKA_FN_HOST_ACC auto setExtent(
            TExtents & extents,
            TExtent const & extent)
        -> void
        {
            traits::SetExtent<
                std::integral_constant<std::size_t, Tidx>,
                TExtents,
                TExtent>
            ::setExtent(
                extents,
                extent);
        }
        //-----------------------------------------------------------------------------
        //! Sets the width.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TExtents,
            typename TExtent>
        ALPAKA_FN_HOST_ACC auto setWidth(
            TExtents & extents,
            TExtent const & width)
        -> void
        {
            setExtent<dim::Dim<TExtents>::value - 1u>(extents, width);
        }
        //-----------------------------------------------------------------------------
        //! Sets the height.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TExtents,
            typename TExtent>
        ALPAKA_FN_HOST_ACC auto setHeight(
            TExtents & extents,
            TExtent const & height)
        -> void
        {
            setExtent<dim::Dim<TExtents>::value - 2u>(extents, height);
        }
        //-----------------------------------------------------------------------------
        //! Sets the depth.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TExtents,
            typename TExtent>
        ALPAKA_FN_HOST_ACC auto setDepth(
            TExtents & extents,
            TExtent const & depth)
        -> void
        {
            setExtent<dim::Dim<TExtents>::value - 3u>(extents, depth);
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
                typename TExtents>
            struct GetExtent<
                std::integral_constant<std::size_t, 0u>,
                TExtents,
                typename std::enable_if<
                    std::is_integral<TExtents>::value>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getExtent(
                    TExtents const & extents)
                -> size::Size<TExtents>
                {
                    return extents;
                }
            };
            //#############################################################################
            //! The unsigned integral width set trait specialization.
            //#############################################################################
            template<
                typename TExtents,
                typename TExtent>
            struct SetExtent<
                std::integral_constant<std::size_t, 0u>,
                TExtents,
                TExtent,
                typename std::enable_if<
                    std::is_integral<TExtents>::value>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto setExtent(
                    TExtents const & extents,
                    TExtent const & extent)
                -> void
                {
                    extents = extent;
                }
            };
        }
    }
}
