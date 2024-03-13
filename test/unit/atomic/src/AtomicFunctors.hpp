/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, Jan Stephan, Bernhard Manfred Gruber,
 * Antonio Di Pilato
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

//! @file This file provides functors where a atomic function and the corresponding operation availble within one
//! object. This allwos testing of the atomic function and atomicOp<Op>() without the usage of function pointers whcih
//! is not suppoted by all backends.

#include <alpaka/atomic/Op.hpp>
#include <alpaka/atomic/Traits.hpp>

#include <utility>

namespace alpaka::test::unit::atomic
{
    struct Add
    {
        using Op = alpaka::AtomicAdd;

        template<typename... TArgs>
        static ALPAKA_FN_ACC auto atomic(TArgs&&... args)
        {
            return alpaka::atomicAdd(std::forward<TArgs>(args)...);
        }
    };

    struct Sub
    {
        using Op = alpaka::AtomicSub;

        template<typename... TArgs>
        static ALPAKA_FN_ACC auto atomic(TArgs&&... args)
        {
            return alpaka::atomicSub(std::forward<TArgs>(args)...);
        }
    };

    struct Min
    {
        using Op = alpaka::AtomicMin;

        template<typename... TArgs>
        static ALPAKA_FN_ACC auto atomic(TArgs&&... args)
        {
            return alpaka::atomicMin(std::forward<TArgs>(args)...);
        }
    };

    struct Max
    {
        using Op = alpaka::AtomicMax;

        template<typename... TArgs>
        static ALPAKA_FN_ACC auto atomic(TArgs&&... args)
        {
            return alpaka::atomicMax(std::forward<TArgs>(args)...);
        }
    };

    struct Exch
    {
        using Op = alpaka::AtomicExch;

        template<typename... TArgs>
        static ALPAKA_FN_ACC auto atomic(TArgs&&... args)
        {
            return alpaka::atomicExch(std::forward<TArgs>(args)...);
        }
    };

    struct Dec
    {
        using Op = alpaka::AtomicDec;

        template<typename... TArgs>
        static ALPAKA_FN_ACC auto atomic(TArgs&&... args)
        {
            return alpaka::atomicDec(std::forward<TArgs>(args)...);
        }
    };

    struct Inc
    {
        using Op = alpaka::AtomicInc;

        template<typename... TArgs>
        static ALPAKA_FN_ACC auto atomic(TArgs&&... args)
        {
            return alpaka::atomicInc(std::forward<TArgs>(args)...);
        }
    };

    struct Cas
    {
        using Op = alpaka::AtomicCas;

        template<typename... TArgs>
        static ALPAKA_FN_ACC auto atomic(TArgs&&... args)
        {
            return alpaka::atomicCas(std::forward<TArgs>(args)...);
        }
    };

    struct Or
    {
        using Op = alpaka::AtomicOr;

        template<typename... TArgs>
        static ALPAKA_FN_ACC auto atomic(TArgs&&... args)
        {
            return alpaka::atomicOr(std::forward<TArgs>(args)...);
        }
    };

    struct Xor
    {
        using Op = alpaka::AtomicXor;

        template<typename... TArgs>
        static ALPAKA_FN_ACC auto atomic(TArgs&&... args)
        {
            return alpaka::atomicXor(std::forward<TArgs>(args)...);
        }
    };

    struct And
    {
        using Op = alpaka::AtomicAnd;

        template<typename... TArgs>
        static ALPAKA_FN_ACC auto atomic(TArgs&&... args)
        {
            return alpaka::atomicAnd(std::forward<TArgs>(args)...);
        }
    };

} // namespace alpaka::test::unit::atomic
