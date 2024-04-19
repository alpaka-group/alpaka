/* Copyright 2022 Benjamin Worpitz, Bert Wesarg, René Widera, Sergei Bastrakov, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cstddef>

namespace alpaka
{
    //! Kernel function attributes struct. Attributes are filled by calling the API of the accelerator using the kernel
    //! function as an argument.
    struct KernelFunctionAttributes
    {
        std::size_t constSizeBytes{0};
        std::size_t localSizeBytes{0};
        std::size_t sharedSizeBytes{0};
        int maxDynamicSharedSizeBytes{0};
        int numRegs{0};
        int ptxVersion{0};
        int maxThreadsPerBlock{0};
    };
} // namespace alpaka
