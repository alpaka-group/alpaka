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
        template<typename TList, template<typename...> class TApplicant>
        struct ApplyImpl;

        template<template<typename...> class TList, template<typename...> class TApplicant, typename... T>
        struct ApplyImpl<TList<T...>, TApplicant>
        {
            using type = TApplicant<T...>;
        };
    } // namespace detail
    template<typename TList, template<typename...> class TApplicant>
    using Apply = typename detail::ApplyImpl<TList, TApplicant>::type;
} // namespace alpaka::meta
