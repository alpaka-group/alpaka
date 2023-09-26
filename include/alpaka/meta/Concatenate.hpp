/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 * SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
 * SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

namespace alpaka::meta
{
    namespace detail
    {
        template<typename... T>
        struct ConcatenateImpl;

        template<typename T>
        struct ConcatenateImpl<T>
        {
            using type = T;
        };

        template<template<typename...> class TList, typename... As, typename... Bs, typename... TRest>
        struct ConcatenateImpl<TList<As...>, TList<Bs...>, TRest...>
        {
            using type = typename ConcatenateImpl<TList<As..., Bs...>, TRest...>::type;
        };
    } // namespace detail

    template<typename... T>
    using Concatenate = typename detail::ConcatenateImpl<T...>::type;
} // namespace alpaka::meta
