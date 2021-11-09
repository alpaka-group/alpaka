/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/math/asin/Traits.hpp>

namespace alpaka
{
    namespace math
    {
        //! The standard library asin, implementation covered by the general template.
        class AsinStdLib : public concepts::Implements<ConceptMathAsin, AsinStdLib>
        {
        };
    } // namespace math
} // namespace alpaka
