/* Copyright 2022 Jeffrey Kelling
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
#ifndef ALPAKA_OFFLOAD_USE_BUILTIN_SHARED_MEM
#    define ALPAKA_OFFLOAD_USE_BUILTIN_SHARED_MEM OFF
#endif
    enum class OffloadUseBuiltInSharedMem : char
    {
        OFF, //! Do not use built-in shared memory facilites, use BlockSharedMemDynMember
        DYN_FIXED, //! Use built-in shared memory, allocate fixed space for dyn smem
        DYN_ALLOC //! Use built-in shared memory, use runtime allocation API
    };

    constexpr OffloadUseBuiltInSharedMem OFFLOAD_USE_BUILTIN_SHARED_MEM
        = OffloadUseBuiltInSharedMem::ALPAKA_OFFLOAD_USE_BUILTIN_SHARED_MEM;
} // namespace alpaka
