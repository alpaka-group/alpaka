/* Copyright 2021 Benjamin Worpitz, Jan Stephan
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "mysqrt.hpp"

// a square root calculation using simple operations
ALPAKA_FN_HOST_ACC ALPAKA_FN_EXTERN auto mysqrt(float x) -> float
{
    if(x <= 0)
    {
        return 0.f;
    }

    float result = x;

    for(int i = 0; i < 100; ++i)
    {
        if(result <= 0)
        {
            result = 0.1f;
        }
        float delta = x - (result * result);
        result = result + 0.5f * delta / result;
    }
    return result;
}
