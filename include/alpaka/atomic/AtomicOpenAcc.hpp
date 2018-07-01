/**
* \file
* Copyright 2018 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#ifdef _OPENACC

#include <alpaka/atomic/Op.hpp>
#include <alpaka/atomic/Traits.hpp>

#include <boost/core/ignore_unused.hpp>

namespace alpaka
{
    namespace atomic
    {
        //#############################################################################
        //! The OpenAcc atomic ops.
        class AtomicOpenAcc
        {
        public:
            using AtomicBase = AtomicOpenAcc;

            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA AtomicOpenAcc() = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA AtomicOpenAcc(AtomicOpenAcc const &) = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA AtomicOpenAcc(AtomicOpenAcc &&) = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA auto operator=(AtomicOpenAcc const &) -> AtomicOpenAcc & = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA auto operator=(AtomicOpenAcc &&) -> AtomicOpenAcc & = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA /*virtual*/ ~AtomicOpenAcc() = default;
        };

        namespace traits
        {
            //-----------------------------------------------------------------------------
            //! The OpenACC atomic Addition operation.
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Add,
                atomic::AtomicOpenAcc,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicOpenAcc const & atomic,
                    T * const addr,
                    T const & value)
                -> T
                {
                    boost::ignore_unused(atomic);
                    T old;
                    #pragma acc atomic capture
                    {
                        old = *addr;
                        *addr += value;
                    }
                    return old;
                }
            };
            //-----------------------------------------------------------------------------
            //! The OpenACC atomic Subtraction operation.
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Sub,
                atomic::AtomicOpenAcc,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicOpenAcc const & atomic,
                    T * const addr,
                    T const & value)
                -> T
                {
                    boost::ignore_unused(atomic);
                    T old;
                    #pragma acc atomic capture
                    {
                        old = *addr;
                        *addr -= value;
                    }
                    return old;
                }
            };
            //-----------------------------------------------------------------------------
            //! The OpenACC atomic MIN operation.
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Min,
                atomic::AtomicOpenAcc,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicOpenAcc const & atomic,
                    T * const addr,
                    T const & value)
                -> T
                {
                    boost::ignore_unused(atomic);
                    T old;
                    #pragma acc atomic capture
                    {
                        old = *addr;
                        *addr = (value < *addr) ? value : *addr;
                    }
                    return old;
                }
            };
            //-----------------------------------------------------------------------------
            //! The OpenACC atomic MAX operation.
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Max,
                atomic::AtomicOpenAcc,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicOpenAcc const & atomic,
                    T * const addr,
                    T const & value)
                -> T
                {
                    boost::ignore_unused(atomic);
                    T old;
                    #pragma acc atomic capture
                    {
                        old = *addr;
                        *addr = (value > *addr) ? value : *addr;
                    }
                    return old;
                }
            };
            //-----------------------------------------------------------------------------
            //! The OpenACC atomic Exchange operation.
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Exch,
                atomic::AtomicOpenAcc,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicOpenAcc const & atomic,
                    T * const addr,
                    T const & value)
                -> T
                {
                    boost::ignore_unused(atomic);
                    T old;
                    #pragma acc atomic capture
                    {
                        old = *addr;
                        *addr = value;
                    }
                    return old;
                }
            };
            //-----------------------------------------------------------------------------
            //! The OpenACC atomic Increment operation.
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Inc,
                atomic::AtomicOpenAcc,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicOpenAcc const & atomic,
                    T * const addr,
                    T const & value)
                -> T
                {
                    boost::ignore_unused(atomic);
                    T old;
                    #pragma acc atomic capture
                    {
                        old = *addr;
                        *addr = ((old >= value) ? 0 : old + 1);
                    }
                    return old;
                }
            };
            //-----------------------------------------------------------------------------
            //! The OpenACC atomic Decrement operation.
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Dec,
                atomic::AtomicOpenAcc,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicOpenAcc const & atomic,
                    T * const addr,
                    T const & value)
                -> T
                {
                    boost::ignore_unused(atomic);
                    T old;
                    #pragma acc atomic capture
                    {
                        old = *addr;
                        *addr = (((old == 0) || (old > value)) ? value : (old - 1));
                    }
                    return old;
                }
            };
            //-----------------------------------------------------------------------------
            //! The OpenACC atomic AND operation.
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::And,
                atomic::AtomicOpenAcc,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicOpenAcc const & atomic,
                    T * const addr,
                    T const & value)
                -> T
                {
                    boost::ignore_unused(atomic);
                    T old;
                    #pragma acc atomic capture
                    {
                        old = *addr;
                        *addr &= value;
                    }
                    return old;
                }
            };
            //-----------------------------------------------------------------------------
            //! The OpenACC atomic OR operation.
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Or,
                atomic::AtomicOpenAcc,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicOpenAcc const & atomic,
                    T * const addr,
                    T const & value)
                -> T
                {
                    boost::ignore_unused(atomic);
                    T old;
                    #pragma acc atomic capture
                    {
                        old = *addr;
                        *addr |= value;
                    }
                    return old;
                }
            };
            //-----------------------------------------------------------------------------
            //! The OpenACC atomic XOR operation.
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Xor,
                atomic::AtomicOpenAcc,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicOpenAcc const & atomic,
                    T * const addr,
                    T const & value)
                -> T
                {
                    boost::ignore_unused(atomic);
                    T old;
                    #pragma acc atomic capture
                    {
                        old = *addr;
                        *addr ^= value;
                    }
                    return old;
                }
            };
            //-----------------------------------------------------------------------------
            //! The OpenACC atomic CAS operation.
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Cas,
                atomic::AtomicOpenAcc,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto atomicOp(
                    atomic::AtomicOpenAcc const & atomic,
                    T * const addr,
                    T const & compare,
                    T const & value)
                -> T
                {
                    boost::ignore_unused(atomic);
                    T old;
                    #pragma acc atomic capture
                    {
                        old = *addr;
                        *addr = ((old == compare) ? value : old);
                    }
                    return old;
                }
            };
        }
    }
}

#endif
