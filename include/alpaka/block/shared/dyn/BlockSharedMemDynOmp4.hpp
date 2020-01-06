/* Copyright 2019 Benjamin Worpitz, René Widera
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

#include <alpaka/block/shared/dyn/Traits.hpp>

#include <type_traits>
#include <array>

namespace alpaka
{
    namespace block
    {
        namespace shared
        {
            namespace dyn
            {
                //#############################################################################
                //! The GPU CUDA block shared memory allocator.
                class BlockSharedMemDynOmp4 : public concepts::Implements<ConceptBlockSharedDyn, BlockSharedMemDynOmp4>
                {
                    mutable std::array<char, 30<<10> m_mem; // ! static 30kB
                    std::size_t m_dynSize;

                public:
                    //-----------------------------------------------------------------------------
                    BlockSharedMemDynOmp4(size_t sizeBytes) : m_dynSize(sizeBytes) {}
                    //-----------------------------------------------------------------------------
                    BlockSharedMemDynOmp4(BlockSharedMemDynOmp4 const &) = delete;
                    //-----------------------------------------------------------------------------
                    BlockSharedMemDynOmp4(BlockSharedMemDynOmp4 &&) = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(BlockSharedMemDynOmp4 const &) -> BlockSharedMemDynOmp4 & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(BlockSharedMemDynOmp4 &&) -> BlockSharedMemDynOmp4 & = delete;
                    //-----------------------------------------------------------------------------
                    /*virtual*/ ~BlockSharedMemDynOmp4() = default;

                    char* dynMemBegin() const {return m_mem.data();}
                    char* staticMemBegin() const {return m_mem.data()+m_dynSize;}
                };

                namespace traits
                {
                    //#############################################################################
                    template<
                        typename T>
                    struct GetMem<
                        T,
                        BlockSharedMemDynOmp4>
                    {
                        //-----------------------------------------------------------------------------
                        static auto getMem(
                            block::shared::dyn::BlockSharedMemDynOmp4 const &mem)
                        -> T *
                        {
                            return reinterpret_cast<T*>(mem.dynMemBegin());
                        }
                    };
                }
            }
        }
    }
}

#endif
