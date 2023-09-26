/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 *
 * SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
 * SPDX-FileContributor: Jeffrey Kelling <j.kelling@hzdr.de>
 * SPDX-FileContributor: Jiří Vyskočil <jiri@vyskocil.com>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <array>
#include <cstdint>

namespace alpaka::rand::engine
{
    /** Philox backend using std::array for Key and Counter storage
     *
     * @tparam TParams Philox algorithm parameters \sa PhiloxParams
     */
    template<typename TParams>
    class PhiloxBaseStdArray
    {
    public:
        using Counter = std::array<std::uint32_t, TParams::counterSize>; ///< Counter type = std::array
        using Key = std::array<std::uint32_t, TParams::counterSize / 2>; ///< Key type = std::array
        template<typename TScalar>
        using ResultContainer
            = std::array<TScalar, TParams::counterSize>; ///< Vector template for distribution results
    };
} // namespace alpaka::rand::engine
