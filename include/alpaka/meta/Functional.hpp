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

#include "alpaka/core/Common.hpp"

namespace alpaka::meta
{
    template<typename T>
    struct min
    {
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC constexpr auto operator()(T const& lhs, T const& rhs) const
        {
            return (lhs < rhs) ? lhs : rhs;
        }
    };

    template<typename T>
    struct max
    {
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC constexpr auto operator()(T const& lhs, T const& rhs) const
        {
            return (lhs > rhs) ? lhs : rhs;
        }
    };
} // namespace alpaka::meta
