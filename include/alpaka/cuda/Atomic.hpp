/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of either the GNU General Public License or
* the GNU Lesser General Public License for more details. as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/interfaces/Atomic.hpp> // IAtomic

//#define ALPAKA_ATOMIC_ADD_FLOAT_CAS   // Undefine this to use the floating point implementation of atomic add using atomicCAS for device with compute capability < 2.0. 
                                        // This should be slower then the version using atomicExch.

namespace alpaka
{
    namespace cuda
    {
        namespace detail
        {
            //#############################################################################
            //! The CUDA accelerator atomic operations.
            //#############################################################################
            class AtomicCuda
            {
            public:
                //-----------------------------------------------------------------------------
                //! Default-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC AtomicCuda() = default;
                //-----------------------------------------------------------------------------
                //! Copy-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC AtomicCuda(AtomicCuda const &) = default;
                //-----------------------------------------------------------------------------
                //! Move-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC AtomicCuda(AtomicCuda &&) = default;
                //-----------------------------------------------------------------------------
                //! Copy-assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC AtomicCuda & operator=(AtomicCuda const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST ~AtomicCuda() noexcept = default;
            };
            using TInterfacedAtomic = alpaka::detail::IAtomic<AtomicCuda>;
        }
    }

    namespace detail
    {
        //#############################################################################
        //! The specializations to execute the requested atomic operations of the CUDA accelerator.
        // See: http://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions how to implement everything with CAS
        //#############################################################################
        //-----------------------------------------------------------------------------
        // Add.
        //-----------------------------------------------------------------------------
        //-----------------------------------------------------------------------------
        //! The CUDA accelerator atomic operation functor.
        //-----------------------------------------------------------------------------
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Add, int>
        {
            ALPAKA_FCT_ACC int operator()(cuda::detail::AtomicCuda const & , int * const addr, int const & value) const
            {
                return atomicAdd(addr, value);
            }
        };
        //-----------------------------------------------------------------------------
        //! The CUDA accelerator atomic operation functor.
        //-----------------------------------------------------------------------------
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Add, unsigned int>
        {
            ALPAKA_FCT_ACC unsigned int operator()(cuda::detail::AtomicCuda const &, unsigned int * const addr, unsigned int const & value) const
            {
                return atomicAdd(addr, value);
            }
        };
        //-----------------------------------------------------------------------------
        //! The CUDA accelerator atomic operation functor.
        //-----------------------------------------------------------------------------
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Add, unsigned long long int>
        {
            ALPAKA_FCT_ACC unsigned long long int operator()(cuda::detail::AtomicCuda const &, unsigned long long int * const addr, unsigned long long int const & value) const
            {
                return atomicAdd(addr, value);
            }
        };
        //-----------------------------------------------------------------------------
        //! The CUDA accelerator atomic operation functor.
        //-----------------------------------------------------------------------------
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Add, float>
        {
            ALPAKA_FCT_ACC float operator()(cuda::detail::AtomicCuda const &, float * const addr, float const & value) const
            {
                return atomicAdd(addr, value);
            }
        };
        //-----------------------------------------------------------------------------
        //! The CUDA accelerator atomic operation functor.
        //-----------------------------------------------------------------------------
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Add, double>
        {
            ALPAKA_FCT_ACC double operator()(cuda::detail::AtomicCuda const &, double * const addr, double const & value) const
            {
                // Code from: http://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions

                unsigned long long int * address_as_ull(reinterpret_cast<unsigned long long int *>(addr));
                unsigned long long int old(*address_as_ull);
                unsigned long long int assumed;
                do
                {
                    assumed = old;
                    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(value + __longlong_as_double(assumed)));
                    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) 
                } 
                while (assumed != old); 
                return __longlong_as_double(old); 
            }
        };
        //-----------------------------------------------------------------------------
        // Sub.
        //-----------------------------------------------------------------------------
        //-----------------------------------------------------------------------------
        //! The CUDA accelerator atomic operation functor.
        //-----------------------------------------------------------------------------
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Sub, int>
        {
            ALPAKA_FCT_ACC int operator()(cuda::detail::AtomicCuda const &, int * const addr, int const & value) const
            {
                return atomicSub(addr, value);
            }
        };
        //-----------------------------------------------------------------------------
        //! The CUDA accelerator atomic operation functor.
        //-----------------------------------------------------------------------------
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Sub, unsigned int>
        {
            ALPAKA_FCT_ACC unsigned int operator()(cuda::detail::AtomicCuda const &, unsigned int * const addr, unsigned int const & value) const
            {
                return atomicSub(addr, value);
            }
        };
        //-----------------------------------------------------------------------------
        // Min.
        //-----------------------------------------------------------------------------
        //-----------------------------------------------------------------------------
        //! The CUDA accelerator atomic operation functor.
        //-----------------------------------------------------------------------------
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Min, int>
        {
            ALPAKA_FCT_ACC int operator()(cuda::detail::AtomicCuda const &, int * const addr, int const & value) const
            {
                return atomicMin(addr, value);
            }
        };
        //-----------------------------------------------------------------------------
        //! The CUDA accelerator atomic operation functor.
        //-----------------------------------------------------------------------------
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Min, unsigned int>
        {
            ALPAKA_FCT_ACC unsigned int operator()(cuda::detail::AtomicCuda const &, unsigned int * const addr, unsigned int const & value) const
            {
                return atomicMin(addr, value);
            }
        };
        //-----------------------------------------------------------------------------
        //! The CUDA accelerator atomic operation functor.
        //-----------------------------------------------------------------------------
        /*template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Min, unsigned long long int>
        {
            ALPAKA_FCT_ACC unsigned long long int operator()(cuda::detail::AtomicCuda const &, unsigned long long int * const addr, unsigned long long int const & value) const
            {
                return atomicMin(addr, value);
            }
        };*/
        //-----------------------------------------------------------------------------
        // Max.
        //-----------------------------------------------------------------------------
        //-----------------------------------------------------------------------------
        //! The CUDA accelerator atomic operation functor.
        //-----------------------------------------------------------------------------
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Max, int>
        {
            ALPAKA_FCT_ACC int operator()(cuda::detail::AtomicCuda const &, int * const addr, int const & value) const
            {
                return atomicMax(addr, value);
            }
        };
        //-----------------------------------------------------------------------------
        //! The CUDA accelerator atomic operation functor.
        //-----------------------------------------------------------------------------
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Max, unsigned int>
        {
            ALPAKA_FCT_ACC unsigned int operator()(cuda::detail::AtomicCuda const &, unsigned int * const addr, unsigned int const & value) const
            {
                return atomicMax(addr, value);
            }
        };
        //-----------------------------------------------------------------------------
        //! The CUDA accelerator atomic operation functor.
        //-----------------------------------------------------------------------------
        /*template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Max, unsigned long long int>
        {
            ALPAKA_FCT_ACC unsigned long long int operator()(cuda::detail::AtomicCuda const &, unsigned long long int * const addr, unsigned long long int const & value) const
            {
                return atomicMax(addr, value);
            }
        };*/
        //-----------------------------------------------------------------------------
        // Exch.
        //-----------------------------------------------------------------------------
        //-----------------------------------------------------------------------------
        //! The CUDA accelerator atomic operation functor.
        //-----------------------------------------------------------------------------
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Exch, int>
        {
            ALPAKA_FCT_ACC int operator()(cuda::detail::AtomicCuda const & , int * const addr, int const & value) const
            {
                return atomicExch(addr, value);
            }
        };
        //-----------------------------------------------------------------------------
        //! The CUDA accelerator atomic operation functor.
        //-----------------------------------------------------------------------------
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Exch, unsigned int>
        {
            ALPAKA_FCT_ACC unsigned int operator()(cuda::detail::AtomicCuda const &, unsigned int * const addr, unsigned int const & value) const
            {
                return atomicExch(addr, value);
            }
        };
        //-----------------------------------------------------------------------------
        //! The CUDA accelerator atomic operation functor.
        //-----------------------------------------------------------------------------
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Exch, unsigned long long int>
        {
            ALPAKA_FCT_ACC unsigned long long int operator()(cuda::detail::AtomicCuda const &, unsigned long long int * const addr, unsigned long long int const & value) const
            {
                return atomicExch(addr, value);
            }
        };
        //-----------------------------------------------------------------------------
        //! The CUDA accelerator atomic operation functor.
        //-----------------------------------------------------------------------------
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Exch, float>
        {
            ALPAKA_FCT_ACC float operator()(cuda::detail::AtomicCuda const &, float * const addr, float const & value) const
            {
                return atomicExch(addr, value);
            }
        };
        //-----------------------------------------------------------------------------
        // Inc.
        //-----------------------------------------------------------------------------
        //-----------------------------------------------------------------------------
        //! The CUDA accelerator atomic operation functor.
        //-----------------------------------------------------------------------------
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Inc, unsigned int>
        {
            ALPAKA_FCT_ACC unsigned int operator()(cuda::detail::AtomicCuda const & , unsigned int * const addr, unsigned int const & value) const
            {
                return atomicInc(addr, value);
            }
        };
        //-----------------------------------------------------------------------------
        // Dec.
        //-----------------------------------------------------------------------------
        //-----------------------------------------------------------------------------
        //! The CUDA accelerator atomic operation functor.
        //-----------------------------------------------------------------------------
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Dec, unsigned int>
        {
            ALPAKA_FCT_ACC unsigned int operator()(cuda::detail::AtomicCuda const & , unsigned int * const addr, unsigned int const & value) const
            {
                return atomicDec(addr, value);
            }
        };
        //-----------------------------------------------------------------------------
        // And.
        //-----------------------------------------------------------------------------
        //-----------------------------------------------------------------------------
        //! The CUDA accelerator atomic operation functor.
        //-----------------------------------------------------------------------------
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, And, int>
        {
            ALPAKA_FCT_ACC int operator()(cuda::detail::AtomicCuda const &, int * const addr, int const & value) const
            {
                return atomicAnd(addr, value);
            }
        };
        //-----------------------------------------------------------------------------
        //! The CUDA accelerator atomic operation functor.
        //-----------------------------------------------------------------------------
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, And, unsigned int>
        {
            ALPAKA_FCT_ACC unsigned int operator()(cuda::detail::AtomicCuda const &, unsigned int * const addr, unsigned int const & value) const
            {
                return atomicAnd(addr, value);
            }
        };
        //-----------------------------------------------------------------------------
        //! The CUDA accelerator atomic operation functor.
        //-----------------------------------------------------------------------------
        /*template<>
        struct AtomicOp<cuda::detail::AtomicCuda, And, unsigned long long int>
        {
            ALPAKA_FCT_ACC unsigned long long int operator()(cuda::detail::AtomicCuda const &, unsigned long long int * const addr, unsigned long long int const & value) const
            {
                return atomicAnd(addr, value);
            }
        };*/
        //-----------------------------------------------------------------------------
        // Or.
        //-----------------------------------------------------------------------------
        //-----------------------------------------------------------------------------
        //! The CUDA accelerator atomic operation functor.
        //-----------------------------------------------------------------------------
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Or, int>
        {
            ALPAKA_FCT_ACC int operator()(cuda::detail::AtomicCuda const &, int * const addr, int const & value) const
            {
                return atomicOr(addr, value);
            }
        };
        //-----------------------------------------------------------------------------
        //! The CUDA accelerator atomic operation functor.
        //-----------------------------------------------------------------------------
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Or, unsigned int>
        {
            ALPAKA_FCT_ACC unsigned int operator()(cuda::detail::AtomicCuda const &, unsigned int * const addr, unsigned int const & value) const
            {
                return atomicOr(addr, value);
            }
        };
        //-----------------------------------------------------------------------------
        //! The CUDA accelerator atomic operation functor.
        //-----------------------------------------------------------------------------
        /*template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Or, unsigned long long int>
        {
            ALPAKA_FCT_ACC unsigned long long int operator()(cuda::detail::AtomicCuda const &, unsigned long long int * const addr, unsigned long long int const & value) const
            {
                return atomicOr(addr, value);
            }
        };*/
        //-----------------------------------------------------------------------------
        // Xor.
        //-----------------------------------------------------------------------------
        //-----------------------------------------------------------------------------
        //! The CUDA accelerator atomic operation functor.
        //-----------------------------------------------------------------------------
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Xor, int>
        {
            ALPAKA_FCT_ACC int operator()(cuda::detail::AtomicCuda const &, int * const addr, int const & value) const
            {
                return atomicXor(addr, value);
            }
        };
        //-----------------------------------------------------------------------------
        //! The CUDA accelerator atomic operation functor.
        //-----------------------------------------------------------------------------
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Xor, unsigned int>
        {
            ALPAKA_FCT_ACC unsigned int operator()(cuda::detail::AtomicCuda const &, unsigned int * const addr, unsigned int const & value) const
            {
                return atomicXor(addr, value);
            }
        };
        //-----------------------------------------------------------------------------
        //! The CUDA accelerator atomic operation functor.
        //-----------------------------------------------------------------------------
        /*template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Xor, unsigned long long int>
        {
            ALPAKA_FCT_ACC unsigned long long int operator()(cuda::detail::AtomicCuda const &, unsigned long long int * const addr, unsigned long long int const & value) const
            {
                return atomicXor(addr, value);
            }
        };*/
        //-----------------------------------------------------------------------------
        // Cas.
        //-----------------------------------------------------------------------------
        //-----------------------------------------------------------------------------
        //! The CUDA accelerator atomic operation functor.
        //-----------------------------------------------------------------------------
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Cas, int>
        {
            ALPAKA_FCT_ACC int operator()(cuda::detail::AtomicCuda const &, int * const addr, int const & compare, int const & value) const
            {
                return atomicCAS(addr, compare, value);
            }
        };
        //-----------------------------------------------------------------------------
        //! The CUDA accelerator atomic operation functor.
        //-----------------------------------------------------------------------------
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Cas, unsigned int>
        {
            ALPAKA_FCT_ACC unsigned int operator()(cuda::detail::AtomicCuda const &, unsigned int * const addr, unsigned int const & compare, unsigned int const & value) const
            {
                return atomicCAS(addr, compare, value);
            }
        };
        //-----------------------------------------------------------------------------
        //! The CUDA accelerator atomic operation functor.
        //-----------------------------------------------------------------------------
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Cas, unsigned long long int>
        {
            ALPAKA_FCT_ACC unsigned long long int operator()(cuda::detail::AtomicCuda const &, unsigned long long int * const addr, unsigned long long int const & compare, unsigned long long int const & value) const
            {
                return atomicCAS(addr, compare, value);
            }
        };
    }
}
