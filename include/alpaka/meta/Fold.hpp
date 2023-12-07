/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 * SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
 * SPDX-FileContributor: Axel Hübl <a.huebl@plasma.ninja>
 * SPDX-FileContributor: Jan Stephan <j.stephan@hzdr.de>
 * SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"

namespace alpaka::meta
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TFnObj, typename T>
    ALPAKA_FN_HOST_ACC constexpr auto foldr(TFnObj const& /* f */, T const& t) -> T
    {
        return t;
    }

    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TFnObj, typename T0, typename T1, typename... Ts>
    ALPAKA_FN_HOST_ACC constexpr auto foldr(TFnObj const& f, T0 const& t0, T1 const& t1, Ts const&... ts)
    {
        return f(t0, foldr(f, t1, ts...));
    }
} // namespace alpaka::meta
