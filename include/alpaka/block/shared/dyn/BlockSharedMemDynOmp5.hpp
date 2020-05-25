/* Copyright 2019 Benjamin Worpitz, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OMP5_ENABLED

#if _OPENMP < 201307
    #error If ALPAKA_ACC_ANY_BT_OMP5_ENABLED is set, the compiler has to support OpenMP 4.0 or higher!
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
                class BlockSharedMemDynOmp5 : public concepts::Implements<ConceptBlockSharedDyn, BlockSharedMemDynOmp5>
                {
                    mutable std::array<char, 30<<10> m_mem; // ! static 30kB
                    std::size_t m_dynSize;

                public:
                    //-----------------------------------------------------------------------------
                    BlockSharedMemDynOmp5(size_t sizeBytes) : m_dynSize(sizeBytes) {}
                    //-----------------------------------------------------------------------------
                    BlockSharedMemDynOmp5(BlockSharedMemDynOmp5 const &) = delete;
                    //-----------------------------------------------------------------------------
                    BlockSharedMemDynOmp5(BlockSharedMemDynOmp5 &&) = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(BlockSharedMemDynOmp5 const &) -> BlockSharedMemDynOmp5 & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(BlockSharedMemDynOmp5 &&) -> BlockSharedMemDynOmp5 & = delete;
                    //-----------------------------------------------------------------------------
                    /*virtual*/ ~BlockSharedMemDynOmp5() = default;

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
                        BlockSharedMemDynOmp5>
                    {
                        //-----------------------------------------------------------------------------
                        static auto getMem(
                            block::shared::dyn::BlockSharedMemDynOmp5 const &mem)
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
