/* Copyright 2022 Jeffrey Kelling
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

namespace alpaka
{
    struct OffloadBuiltInSharedMemOff
    {
    };
    struct OffloadBuiltInSharedMemFixed
    {
    };
    struct OffloadBuiltInSharedMemAlloc
    {
    };
#ifdef ALPAKA_OFFLOAD_BUILTIN_SHARED_MEM_FIXED
    using OffloadBuiltInSharedMem = OffloadBuiltInSharedMemFixed;
#elif defined(ALPAKA_OFFLOAD_BUILTIN_SHARED_MEM_ALLOC)
    using OffloadBuiltInSharedMem = OffloadBuiltInSharedMemAlloc;
#else
    using OffloadBuiltInSharedMem = OffloadBuiltInSharedMemOff;
#endif
} // namespace alpaka
