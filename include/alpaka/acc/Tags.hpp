/* Copyright 2022 René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

namespace alpaka
{
    struct ConceptAccCpuFiber;
    struct ConceptAccCpuOmp2Blocks;
    struct ConceptAccCpuOmp2Threads;
    struct ConceptAccCpuSerial;
    struct ConceptAccCpuSyclIntel;

    struct ConceptTaskKernelCpuTbbBlocks;
    struct ConceptAccCpuThreads;
    struct ConceptAccFpgaSyclIntel;
    struct ConceptAccFpgaSyclXilinx;
    struct ConceptAccCudaRt;
    struct ConceptAccHipRt;

    struct ConceptAccGenericSycl;
    struct ConceptAccGpuSyclIntel;

    struct ConceptAccOacc;
    struct ConceptAccOmp5;
} // namespace alpaka
