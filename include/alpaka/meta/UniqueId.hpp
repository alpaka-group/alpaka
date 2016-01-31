/**
* \file
* Copyright 2015 Benjamin Worpitz, Rene Widera
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

#include <boost/predef.h>           // workarounds

#include <cstddef>                  // std::size_t

namespace alpaka
{
    namespace meta
    {
        namespace detail
        {
            static std::size_t constexpr maxUniqueId = 128u;

            //#############################################################################
            // Based on the code from Filip Roseen at http://b.atch.se/posts/constexpr-counter/
            //#############################################################################
            template<
                std::size_t N>
            struct flag;

            //-----------------------------------------------------------------------------
            //
            //-----------------------------------------------------------------------------
            template<
                std::size_t N>
            std::size_t constexpr adl_flag(flag<N>);

            //#############################################################################
            //
            //#############################################################################
            template<
                std::size_t N>
            struct flag
            {
    // Declaring the non-template friend function is desired.
    // Therefore, we can disable the warning
#if BOOST_COMP_GNUC
    #pragma GCC diagnostic push
    // warning: friend declaration ‘constexpr std::size_t alpaka::meta::detail::adl_flag(alpaka::meta::detail::flag<N>)’ declares a non-template function [-Wnon-template-friend]
    #pragma GCC diagnostic ignored "-Wnon-template-friend"
#elif BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wundefined-inline"
#elif BOOST_COMP_MSVC
    #pragma warning(push)
    //#pragma warning(disable: 4244)
#endif
                friend std::size_t constexpr adl_flag(flag<N>);
#if BOOST_COMP_GNUC
    #pragma GCC diagnostic pop
#elif BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#elif BOOST_COMP_MSVC
    #pragma warning(pop)
#endif
            };

            //#############################################################################
            //
            //#############################################################################
            template<
                std::size_t N>
            struct writer
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                friend std::size_t constexpr adl_flag(flag<N>)
                {
                    return N;
                }

                static std::size_t constexpr value = N;
            };

#if (BOOST_COMP_MSVC || __CUDACC__)
            //-----------------------------------------------------------------------------
            //! The matcher.
            //-----------------------------------------------------------------------------
            template<
                std::size_t N,
                class = char[noexcept(adl_flag(flag<N>{})) ? +1 : -1]>
            auto constexpr reader(std::size_t, flag<N>)
            -> std::size_t
            {
              return N;
            }
#else
            //-----------------------------------------------------------------------------
            //! The matcher.
            //-----------------------------------------------------------------------------
            template<
                std::size_t N,
                std::size_t = adl_flag(flag<N>{})>
            auto constexpr reader(
                std::size_t,
                flag<N>)
            -> std::size_t
            {
                return N;
            }
#endif
            //-----------------------------------------------------------------------------
            //! The searcher.
            //-----------------------------------------------------------------------------
            template<
                std::size_t N>
            auto constexpr reader(
                float,
                flag<N>,
                std::size_t R = reader(std::size_t{0u}, flag<N-1u>{}))
            -> std::size_t
            {
                return R;
            }
            //-----------------------------------------------------------------------------
            //! Reader base case.
            //-----------------------------------------------------------------------------
            std::size_t constexpr reader(float, flag<0u>)
            {
                return 0u;
            }
        }

        // NOTE: We can not hide this uniqueId implementation inside the detail namespace and forward to it from outside.
        // Compilers would not be forced to reevaluate the default template parameters in this case and will always generate the same ID.
#if (BOOST_COMP_MSVC)
        //-----------------------------------------------------------------------------
        //! \return An unique compile time ID.
        //-----------------------------------------------------------------------------
        template<
            std::size_t N = 1u,
            std::size_t C = detail::reader(std::size_t{0u}, detail::flag<detail::maxUniqueId>{})>
        auto constexpr uniqueId(
            std::size_t R = detail::writer<C + N>::value)
        -> std::size_t
        {
            return R;
        }
#else
        //-----------------------------------------------------------------------------
        //! \return An unique compile time ID.
        //-----------------------------------------------------------------------------
        template<
            std::size_t N = 1u>
        auto constexpr uniqueId(
            std::size_t R = detail::writer<detail::reader(std::size_t{0u}, detail::flag<detail::maxUniqueId>{}) + N>::value)
        -> std::size_t
        {
            return R;
        }
#endif
    }
}