/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 * SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-FileContributor: Simeon Ehrig <s.ehrig@hzdr.de>
 * SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
 * SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once
#include "alpaka/alpaka.hpp"

#include <cstddef>

namespace alpaka::test
{
    template<typename TType, size_t TSize>
    struct Array
    {
        TType m_data[TSize];

        template<typename T_Idx>
        ALPAKA_FN_HOST_ACC auto operator[](const T_Idx idx) const -> TType const&
        {
            return m_data[idx];
        }

        template<typename TIdx>
        ALPAKA_FN_HOST_ACC auto operator[](const TIdx idx) -> TType&
        {
            return m_data[idx];
        }
    };
} // namespace alpaka::test
