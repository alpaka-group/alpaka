/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 *
 * SPDX-FileContributor: Sergei Bastrakov <s.bastrakov@hzdr.de>
 * SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
 * SPDX-FileContributor: Jeffrey Kelling <j.kelling@hzdr.de>
 * SPDX-FileContributor: Jan Stephan <j.stephan@hzdr.de>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/BoostPredef.hpp"

#include <type_traits>

namespace alpaka
{
    //! Provides a decaying wrapper around std::is_same. Example: is_decayed_v<volatile float, float> returns true.
    template<typename T, typename U>
    inline constexpr auto is_decayed_v = std::is_same_v<std::decay_t<T>, std::decay_t<U>>;
} // namespace alpaka
