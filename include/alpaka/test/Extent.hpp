/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 * SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-FileContributor: Simeon Ehrig <s.ehrig@hzdr.de>
 * SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
 * SPDX-FileContributor: Matthias Werner <Matthias.Werner1@tu-dresden.de>
 * SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/alpaka.hpp"

#include <cstddef>

namespace alpaka::test
{
    template<typename TDim, typename TVal>
    inline constexpr auto extentBuf = []
    {
        Vec<TDim, TVal> v;
        if constexpr(TDim::value > 0)
            for(TVal i = 0; i < TVal{TDim::value}; i++)
                v[i] = 11 - i;
        return v;
    }();

    template<typename TDim, typename TVal>
    inline constexpr auto extentSubView = []
    {
        Vec<TDim, TVal> v;
        if constexpr(TDim::value > 0)
            for(TVal i = 0; i < TVal{TDim::value}; i++)
                v[i] = 8 - i * 2;
        return v;
    }();

    template<typename TDim, typename TVal>
    inline constexpr auto offset = []
    {
        Vec<TDim, TVal> v;
        if constexpr(TDim::value > 0)
            for(TVal i = 0; i < TVal{TDim::value}; i++)
                v[i] = 2 + i;
        return v;
    }();
} // namespace alpaka::test
