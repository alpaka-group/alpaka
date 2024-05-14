/* Copyright 2024 Bernhard Manfred Gruber, Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <tuple>
#include <type_traits>

namespace alpaka::meta
{
    namespace detail
    {
        template<typename List>
        struct Front
        {
        };

        template<template<typename...> class List, typename Head, typename... Tail>
        struct Front<List<Head, Tail...>>
        {
            using type = Head;
        };
    } // namespace detail

    template<typename List>
    using Front = typename detail::Front<List>::type;

    template<typename List, typename Value>
    struct Contains : std::false_type
    {
    };

    template<template<typename...> class List, typename Head, typename... Tail, typename Value>
    struct Contains<List<Head, Tail...>, Value>
    {
        static constexpr bool value = std::is_same_v<Head, Value> || Contains<List<Tail...>, Value>::value;
    };

    // copied from https://stackoverflow.com/a/51073558/22035743
    template<typename T>
    struct IsTuple : std::false_type
    {
    };

    template<typename... U>
    struct IsTuple<std::tuple<U...>> : std::true_type
    {
    };

    template<typename T>
    constexpr bool isTuple()
    {
        return IsTuple<std::decay_t<T>>::value;
    }

    template<typename... T>
    struct toTuple
    {
        using type = std::tuple<T...>;
    };

    template<typename... U>
    struct toTuple<std::tuple<U...>>
    {
        using type = std::tuple<U...>;
    };

} // namespace alpaka::meta
