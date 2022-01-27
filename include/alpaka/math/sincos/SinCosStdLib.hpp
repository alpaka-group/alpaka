/* Copyright 2021 Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/math/sincos/Traits.hpp>

namespace alpaka
{
    namespace math
    {
        //! The standard library sincos, implementation covered by the general template.
        class SinCosStdLib : public concepts::Implements<ConceptMathSinCos, SinCosStdLib>
        {
        };
    } // namespace math
} // namespace alpaka
