/* Copyright 2020 Jeffrey Kelling
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef _OPENACC

#include <alpaka/atomic/Traits.hpp>
#include <alpaka/atomic/Op.hpp>

namespace alpaka
{
    //#############################################################################
    //! The OpenACC accelerator's atomic ops.
    //
    //  Atomics can be used in the blocks and threads hierarchy levels.
    //  Atomics are not guaranteed to be safe between devices or grids.
    class AtomicOaccBuiltIn
    {
    public:
        //-----------------------------------------------------------------------------
        AtomicOaccBuiltIn() = default;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST_ACC AtomicOaccBuiltIn(AtomicOaccBuiltIn const &) = delete;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST_ACC AtomicOaccBuiltIn(AtomicOaccBuiltIn &&) = delete;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST_ACC auto operator=(AtomicOaccBuiltIn const &) -> AtomicOaccBuiltIn & = delete;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST_ACC auto operator=(AtomicOaccBuiltIn &&) -> AtomicOaccBuiltIn & = delete;
        //-----------------------------------------------------------------------------
        /*virtual*/ ~AtomicOaccBuiltIn() = default;
    };

    namespace traits
    {

// "omp atomic update capture" is not supported before OpenACC 2.5 and by PGI
// "omp atomic capture {}" works for PGI and GCC, using this even though non-standart
// #if _OPENACC >= 201510

        //#############################################################################
        //! The OpenACC accelerators atomic operation: ADD
        template<
            typename T,
            typename THierarchy>
        struct AtomicOp<
            AtomicAdd,
            AtomicOaccBuiltIn,
            T,
            THierarchy>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST_ACC static auto atomicOp(
                AtomicOaccBuiltIn const &,
                T * const addr,
                T const & value)
            -> T
            {
                T old;
                auto & ref(*addr);
                // atomically update ref, but capture the original value in old
                #pragma acc atomic capture
                {
                    old = ref;
                    ref += value;
                }
                return old;
            }
        };

        //#############################################################################
        //! The OpenACC accelerators atomic operation: SUB
        template<
            typename T,
            typename THierarchy>
        struct AtomicOp<
            AtomicSub,
            AtomicOaccBuiltIn,
            T,
            THierarchy>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST_ACC static auto atomicOp(
                AtomicOaccBuiltIn const &,
                T * const addr,
                T const & value)
            -> T
            {
                T old;
                auto & ref(*addr);
                // atomically update ref, but capture the original value in old
                #pragma acc atomic capture
                {
                    old = ref;
                    ref -= value;
                }
                return old;
            }
        };

        //#############################################################################
        //! The OpenACC accelerators atomic operation: EXCH
        template<
            typename T,
            typename THierarchy>
        struct AtomicOp<
            AtomicExch,
            AtomicOaccBuiltIn,
            T,
            THierarchy>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST_ACC static auto atomicOp(
                AtomicOaccBuiltIn const &,
                T * const addr,
                T const & value)
            -> T
            {
                T old;
                auto & ref(*addr);
                // atomically update ref, but capture the original value in old
#if !BOOST_COMP_PGI || defined TPR28628 // triggers PGI TPR28628, not atomic until fixed
                #pragma acc atomic capture
#endif
                {
                    old = ref;
                    ref = value;
                }
                return old;
            }
        };

        //#############################################################################
        //! The OpenACC accelerators atomic operation: AND
        template<
            typename T,
            typename THierarchy>
        struct AtomicOp<
            AtomicAnd,
            AtomicOaccBuiltIn,
            T,
            THierarchy>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST_ACC static auto atomicOp(
                AtomicOaccBuiltIn const &,
                T * const addr,
                T const & value)
            -> T
            {
                T old;
                auto & ref(*addr);
                // atomically update ref, but capture the original value in old
                #pragma acc atomic capture
                {
                    old = ref;
                    ref &= value;
                }
                return old;
            }
        };

        //#############################################################################
        //! The OpenACC accelerators atomic operation: OR
        template<
            typename T,
            typename THierarchy>
        struct AtomicOp<
            AtomicOr,
            AtomicOaccBuiltIn,
            T,
            THierarchy>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST_ACC static auto atomicOp(
                AtomicOaccBuiltIn const &,
                T * const addr,
                T const & value)
            -> T
            {
                T old;
                auto & ref(*addr);
                // atomically update ref, but capture the original value in old
                #pragma acc atomic capture
                {
                    old = ref;
                    ref |= value;
                }
                return old;
            }
        };

        //#############################################################################
        //! The OpenACC accelerators atomic operation: XOR
        template<
            typename T,
            typename THierarchy>
        struct AtomicOp<
            AtomicXor,
            AtomicOaccBuiltIn,
            T,
            THierarchy>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST_ACC static auto atomicOp(
                AtomicOaccBuiltIn const &,
                T * const addr,
                T const & value)
            -> T
            {
                T old;
                auto & ref(*addr);
                // atomically update ref, but capture the original value in old
                #pragma acc atomic capture
                {
                    old = ref;
                    ref ^= value;
                }
                return old;
            }
        };

        //#############################################################################
        //! The OpenACC accelerators atomic operation: Min
        template<
            typename T,
            typename THierarchy>
        struct AtomicOp<
            AtomicMin,
            AtomicOaccBuiltIn,
            T,
            THierarchy>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST_ACC static auto atomicOp(
                AtomicOaccBuiltIn const &,
                T * const addr,
                T const & value)
            -> T
            {
                T old;
                auto & ref(*addr);
                // atomically update ref, but capture the original value in old
                #pragma acc atomic capture
                {
                    old = ref;
                    ref = (ref <= value) ? ref : value;
                }
                return old;
            }
        };

        //#############################################################################
        //! The OpenACC accelerators atomic operation: Max
        template<
            typename T,
            typename THierarchy>
        struct AtomicOp<
            AtomicMax,
            AtomicOaccBuiltIn,
            T,
            THierarchy>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST_ACC static auto atomicOp(
                AtomicOaccBuiltIn const &,
                T * const addr,
                T const & value)
            -> T
            {
                T old;
                auto & ref(*addr);
                // atomically update ref, but capture the original value in old
                #pragma acc atomic capture
                {
                    old = ref;
                    ref = (ref >= value) ? ref : value;
                }
                return old;
            }
        };

        //#############################################################################
        //! The OpenACC accelerators atomic operation: Inc
        template<
            typename T,
            typename THierarchy>
        struct AtomicOp<
            AtomicInc,
            AtomicOaccBuiltIn,
            T,
            THierarchy>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST_ACC static auto atomicOp(
                AtomicOaccBuiltIn const &,
                T * const addr,
                T const & value)
            -> T
            {
                T old;
                auto & ref(*addr);
                // atomically update ref, but capture the original value in old
                #pragma acc atomic capture
                {
                    old = ref;
                    ref = ((ref >= value) ? 0 : (ref + 1));
                }
                return old;
            }
        };

        //#############################################################################
        //! The OpenACC accelerators atomic operation: Dec
        template<
            typename T,
            typename THierarchy>
        struct AtomicOp<
            AtomicDec,
            AtomicOaccBuiltIn,
            T,
            THierarchy>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST_ACC static auto atomicOp(
                AtomicOaccBuiltIn const &,
                T * const addr,
                T const & value)
            -> T
            {
                T old;
                auto & ref(*addr);
                // atomically update ref, but capture the original value in old
                #pragma acc atomic capture
                {
                    old = ref;
                    ref = ((ref == 0) || (ref > value)) ? value : (ref - 1);
                }
                return old;
            }
        };

        //#############################################################################
        //! The OpenACC accelerators atomic operation: Cas
        template<
            typename T,
            typename THierarchy>
        struct AtomicOp<
            AtomicCas,
            AtomicOaccBuiltIn,
            T,
            THierarchy>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST_ACC static auto atomicOp(
                AtomicOaccBuiltIn const &,
                T * const addr,
                T const & compare,
                T const & value)
            -> T
            {
                T old;
                auto & ref(*addr);
                // atomically update ref, but capture the original value in old
                #pragma acc atomic capture
                {
                    old = ref;
                    ref = (ref == compare ? value : ref);
                }
                return old;
            }
        };
// #endif
    }
}

#endif
