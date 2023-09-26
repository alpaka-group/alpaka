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

#include "alpaka/meta/Integral.hpp"

#include <algorithm>
#include <limits>

namespace alpaka::core
{
    //! \return The input casted and clipped to T.
    template<typename T, typename V>
    auto clipCast(V const& val) -> T
    {
        static_assert(
            std::is_integral_v<T> && std::is_integral_v<V>,
            "clipCast can not be called with non-integral types!");

        constexpr auto max = static_cast<V>(std::numeric_limits<alpaka::meta::LowerMax<T, V>>::max());
        constexpr auto min = static_cast<V>(std::numeric_limits<alpaka::meta::HigherMin<T, V>>::min());

        return static_cast<T>(std::max(min, std::min(max, val)));
    }
} // namespace alpaka::core
