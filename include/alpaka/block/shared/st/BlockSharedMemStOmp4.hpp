/* Copyright 2019 Benjamin Worpitz, Erik Zenker, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_CPU_BT_OMP4_ENABLED

#if _OPENMP < 201307
    #error If ALPAKA_ACC_CPU_BT_OMP4_ENABLED is set, the compiler has to support OpenMP 4.0 or higher!
#endif

#include <alpaka/block/shared/st/Traits.hpp>

#include <type_traits>
#include <cstdint>
#include <omp.h>
#include <array>

namespace alpaka
{
    namespace block
    {
        namespace shared
        {
            namespace st
            {
                //#############################################################################
                //! The GPU CUDA block shared memory allocator.
                class BlockSharedMemStOmp4
                {
                    mutable unsigned int m_allocdBytes = 0;
                    mutable std::array<char, 32<<10> m_mem; // ! static 32kB

                public:
                    using BlockSharedMemStBase = BlockSharedMemStOmp4;

                    //-----------------------------------------------------------------------------
                    BlockSharedMemStOmp4() = default;
                    //-----------------------------------------------------------------------------
                    BlockSharedMemStOmp4(BlockSharedMemStOmp4 const &) = delete;
                    //-----------------------------------------------------------------------------
                    BlockSharedMemStOmp4(BlockSharedMemStOmp4 &&) = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(BlockSharedMemStOmp4 const &) -> BlockSharedMemStOmp4 & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(BlockSharedMemStOmp4 &&) -> BlockSharedMemStOmp4 & = delete;
                    //-----------------------------------------------------------------------------
                    /*virtual*/ ~BlockSharedMemStOmp4() = default;

                    template<class T>
                    T& alloc() const
                    {
                        if(omp_get_thread_num() == 0)
                        {
                            char* buf = &m_mem[m_allocdBytes];
                            new (buf) T();
                            m_allocdBytes += sizeof(T);
                        }
                        #pragma omp barrier
                        return *reinterpret_cast<T*>(&m_mem[m_allocdBytes-sizeof(T)]);
                    }
                };

                namespace traits
                {
                    //#############################################################################
                    template<
                        typename T,
                        std::size_t TuniqueId>
                    struct AllocVar<
                        T,
                        TuniqueId,
                        BlockSharedMemStOmp4>
                    {
                        //-----------------------------------------------------------------------------
                        static auto allocVar(
                            block::shared::st::BlockSharedMemStOmp4 const &smem)
                        -> T &
                        {
                            return *smem.alloc<T>();
                        }
                    };
                    //#############################################################################
                    template<>
                    struct FreeMem<
                        BlockSharedMemStOmp4>
                    {
                        //-----------------------------------------------------------------------------
                        static auto freeMem(
                            block::shared::st::BlockSharedMemStOmp4 const &)
                        -> void
                        {
                            // Nothing to do. CUDA block shared memory is automatically freed when all threads left the block.
                        }
                    };
                }
            }
        }
    }
}

#endif
