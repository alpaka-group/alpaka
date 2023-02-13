/* Copyright 2019 Benjamin Worpitz
 * SPDX-License-Identifier: MPL-2.0
 */

#include "mysqrt.hpp"

// a square root calculation using simple operations
ALPAKA_FN_HOST_ACC auto mysqrt(double x) -> double
{
    if(x <= 0)
    {
        return 0.0;
    }

    double result = x;

    for(int i = 0; i < 100; ++i)
    {
        if(result <= 0)
        {
            result = 0.1;
        }
        double delta = x - (result * result);
        result = result + 0.5 * delta / result;
    }
    return result;
}
