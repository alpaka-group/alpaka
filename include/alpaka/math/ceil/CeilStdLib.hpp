/* Copyright 2021 Axel Huebl, Benjamin Worpitz, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/math/ceil/Traits.hpp>

namespace alpaka
{
    namespace math
    {
        //! The standard library ceil, implementation covered by the general template.
        class CeilStdLib : public concepts::Implements<ConceptMathCeil, CeilStdLib>
        {
        };
    } // namespace math
} // namespace alpaka
