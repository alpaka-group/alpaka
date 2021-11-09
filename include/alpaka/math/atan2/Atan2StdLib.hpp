/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/math/atan2/Traits.hpp>

namespace alpaka
{
    namespace math
    {
        //! The standard library atan2, implementation covered by the general template.
        class Atan2StdLib : public concepts::Implements<ConceptMathAtan2, Atan2StdLib>
        {
        };
    } // namespace math
} // namespace alpaka
