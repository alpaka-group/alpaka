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
        template<typename Ts, template<typename...> class TOp>
        struct TransformImpl;

        template<template<typename...> class TList, typename... Ts, template<typename...> class TOp>
        struct TransformImpl<TList<Ts...>, TOp>
        {
            using type = TList<TOp<Ts>...>;
        };
    } // namespace detail
    template<typename Ts, template<typename...> class TOp>
    using Transform = typename detail::TransformImpl<Ts, TOp>::type;
} // namespace alpaka::meta
