/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#include <tuple>

#include <boost/mpl/sequence_tag.hpp>
#include <boost/mpl/pop_front_fwd.hpp>
#include <boost/mpl/push_front_fwd.hpp>
#include <boost/mpl/push_back_fwd.hpp>
#include <boost/mpl/front_fwd.hpp>
#include <boost/mpl/empty_fwd.hpp>
#include <boost/mpl/size_fwd.hpp>
#include <boost/mpl/at_fwd.hpp>
#include <boost/mpl/back_fwd.hpp>
#include <boost/mpl/clear_fwd.hpp>
#include <boost/mpl/pop_back_fwd.hpp>
#include <boost/mpl/iterator_tags.hpp>
#include <boost/mpl/next_prior.hpp>
#include <boost/mpl/deref.hpp>
#include <boost/mpl/begin_end_fwd.hpp>
// The following definitions specialize the boost::mpl templates that are necessary
// to use a std::tuple with boost::mpl algorithms requiring a type sequence.
// This code is based on:
// http://stackoverflow.com/questions/5099429/how-to-use-stdtuple-types-with-boostmpl-algorithms/15865204#15865204
//#############################################################################

namespace boost
{
    namespace mpl
    {
        namespace aux
        {
            //#############################################################################
            struct std_tuple;
        }

        //#############################################################################
        template<
            typename... TArgs>
        struct sequence_tag<
            std::tuple<TArgs...>>
        {
            using type = aux::std_tuple;
        };
        //#############################################################################
        template<>
        struct front_impl<
            aux::std_tuple>
        {
            //#############################################################################
            template<
                typename TTuple>
            struct apply :
                std::tuple_element<0, TTuple>
            {
            };
        };
        //#############################################################################
        template<>
        struct empty_impl<
            aux::std_tuple>
        {
            //#############################################################################
            template<
                typename TTuple>
            struct apply :
                std::integral_constant<bool, std::tuple_size<TTuple>::value == 0>
            {
            };
        };
        //#############################################################################
        template<>
        struct pop_front_impl<
            aux::std_tuple>
        {
            //#############################################################################
            template<
                typename TTuple>
            struct apply;
            //#############################################################################
            template<
                typename First,
                typename ... Types>
            struct apply<
                std::tuple<First, Types...>>
            {
                using type = std::tuple<Types...>;
            };
        };
        //#############################################################################
        template<>
        struct push_front_impl<
            aux::std_tuple>
        {
            //#############################################################################
            template<
                typename TTuple,
                typename T>
            struct apply;
            //#############################################################################
            template<
                typename T,
                typename ... Args >
            struct apply<
                std::tuple<Args...>, T>
            {
                using type = std::tuple<T, Args...>;
            };
        };
        //#############################################################################
        template<>
        struct push_back_impl<
            aux::std_tuple>
        {
            //#############################################################################
            template<
                typename TTuple,
                typename T>
            struct apply;
            //#############################################################################
            template<
                typename T,
                typename ... TArgs>
            struct apply<
                std::tuple<TArgs...>,
                T>
            {
                using type = std::tuple<TArgs..., T>;
            };
        };
        //#############################################################################
        template<>
        struct size_impl<
            aux::std_tuple>
        {
            //#############################################################################
            template<
                typename TTuple>
            struct apply :
                std::tuple_size<TTuple>
            {
            };
        };
        //#############################################################################
        template<>
        struct at_impl<
            aux::std_tuple>
        {
            //#############################################################################
            template<
                typename TTuple,
                typename N>
            struct apply :
                std::tuple_element<N::value, TTuple>
            {
            };
        };
        //#############################################################################
        template<>
        struct back_impl<
            aux::std_tuple>
        {
            //#############################################################################
            template<
                typename TTuple>
            struct apply :
                std::tuple_element<std::tuple_size<TTuple>::value - 1, TTuple>
            {
            };
        };
        //#############################################################################
        template<>
        struct clear_impl<
            aux::std_tuple>
        {
            //#############################################################################
            template<
                typename TTuple>
            struct apply
            {
                using type = std::tuple<>;
            };
        };
        //#############################################################################
        template<>
        struct pop_back_impl<
            aux::std_tuple>
        {
            //#############################################################################
            template<
                std::size_t ...>
            struct tuple_seq
            {};
            //#############################################################################
            template<
                std::size_t N,
                std::size_t ...S>
            struct tuple_gens :
                tuple_gens<N-1, N-1, S...>
            {};
            //#############################################################################
            template<
                std::size_t ...S>
            struct tuple_gens<0, S...>
            {
                using type = tuple_seq<S...>;
            };
            //#############################################################################
            template<
                typename Tuple,
                typename Index>
            struct apply_impl;
            //#############################################################################
            template<
                typename Tuple,
                std::size_t ... S>
            struct apply_impl<Tuple, tuple_seq<S...>>
            {
                using type = std::tuple<typename std::tuple_element<S, Tuple>::type...>;
            };
            //#############################################################################
            template<
                typename Tuple>
            struct apply :
                apply_impl<Tuple, typename tuple_gens<std::tuple_size<Tuple>::value - 1>::type>
            {};
        };
        //#############################################################################
        template<
            typename... TArgs>
        struct tuple_iter;
        //#############################################################################
        template<
            typename... TArgs>
        struct tuple_iter<
            std::tuple<TArgs...>>
        {
            using tag = aux::std_tuple;
            using category = forward_iterator_tag;
        };
        //#############################################################################
        template<>
        struct begin_impl<
            aux::std_tuple>
        {
            //#############################################################################
            template<
                typename TTuple>
            struct apply
            {
                using type = tuple_iter<TTuple>;
            };
        };
        //#############################################################################
        template<>
        struct end_impl<
            aux::std_tuple>
        {
            //#############################################################################
            template<
                typename TTuple>
            struct apply
            {
                using type = tuple_iter<std::tuple<>>;
            };
        };
        //#############################################################################
        template<
            typename TFirst,
            typename ... TArgs>
        struct deref<
            tuple_iter<std::tuple<TFirst, TArgs...>>>
        {
            using type = TFirst;
        };
        //#############################################################################
        template<
            typename TFirst,
            typename ... TArgs>
        struct next<
            tuple_iter<std::tuple<TFirst, TArgs...>>>
        {
            using type = tuple_iter<std::tuple<TArgs...>>;
        };
    }
}
