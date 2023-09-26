/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 * SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
 * SPDX-FileContributor: Axel Hübl <a.huebl@plasma.ninja>
 * SPDX-FileContributor: Jan Stephan <j.stephan@hzdr.de>
 * SPDX-FileContributor: Matthias Werner <Matthias.Werner1@tu-dresden.de>
 * SPDX-FileContributor: René Widera <r.widera@hzdr.de>
 * SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/atomic/Traits.hpp"

namespace alpaka
{
    //! The NoOp atomic ops.
    class AtomicNoOp
    {
    };

    namespace trait
    {
        //! The NoOp atomic operation.
        template<typename TOp, typename T, typename THierarchy>
        struct AtomicOp<TOp, AtomicNoOp, T, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(AtomicNoOp const& /* atomic */, T* const addr, T const& value) -> T
            {
                return TOp()(addr, value);
            }

            ALPAKA_FN_HOST static auto atomicOp(
                AtomicNoOp const& /* atomic */,
                T* const addr,
                T const& compare,
                T const& value) -> T
            {
                return TOp()(addr, compare, value);
            }
        };
    } // namespace trait
} // namespace alpaka
