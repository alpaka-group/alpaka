/**
* Copyright 2014 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of either the GNU General Public License or
* the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License and the GNU Lesser General Public License
* for more details.
*
* You should have received a copy of the GNU General Public License
* and the GNU Lesser General Public License along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/interfaces/Atomic.hpp> // IAtomic

//#define ALPAKA_ATOMIC_ADD_FLOAT_CAS   // Undefine this to use the floating point implementation of atomic using atomicCAS for device with compute capability <2.0. 
                                        // This should be slower then the version using atomicExch.

namespace alpaka
{
    namespace cuda
    {
        namespace detail
        {
            //#############################################################################
            //! This class holds the implementation details for the atomic operations of the CUDA accelerator.
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
                //! Assignment-operator.
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
        //! Add.
        //-----------------------------------------------------------------------------
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Add, int>
        {
            ALPAKA_FCT_ACC int operator()(cuda::detail::AtomicCuda const & , int * const addr, int const & value) const
            {
                return atomicAdd(addr, value);
            }
        };
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Add, unsigned int>
        {
            ALPAKA_FCT_ACC unsigned int operator()(cuda::detail::AtomicCuda const &, unsigned int * const addr, unsigned int const & value) const
            {
                return atomicAdd(addr, value);
            }
        };
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Add, unsigned long long int>
        {
            ALPAKA_FCT_ACC unsigned long long int operator()(cuda::detail::AtomicCuda const &, unsigned long long int * const addr, unsigned long long int const & value) const
            {
                return atomicAdd(addr, value);
            }
        };
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Add, float>
        {
            ALPAKA_FCT_ACC float operator()(cuda::detail::AtomicCuda const &, float * const addr, float const & value) const
            {
#if __CUDA_ARCH__ >= 200 // for Fermi, atomicAdd supports floats
                return atomicAdd(addr, value);
#else
                // Code adapted from the double version from: http://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions
                // TODO: look at: http://forums.nvidia.com/index.php?showtopic=158039&view=findpost&p=991561 there may be a faster version using Exch

#ifdef ALPAKA_ATOMIC_ADD_FLOAT_CAS
                int * address_as_i(reinterpret_cast<int *>(addr));
                int old(*address_as_i);
                int assumed;
                do
                {
                    assumed = old;
                    old = atomicCAS(address_as_i, assumed, __float_as_int(value + __int_as_float(assumed)));
                    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) 
                } while(assumed != old);
                return __int_as_float(old);
#else
                float ret = atomicExch(addr, 0.0f);
                float old = ret+value;

                while((old = atomicExch(addr, old))!=0.0f)
                {
                    old = atomicExch(addr, 0.0f)+old;
                }

                return ret;
#endif
#endif
            }
        };
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
        //! Sub.
        //-----------------------------------------------------------------------------
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Sub, int>
        {
            ALPAKA_FCT_ACC int operator()(cuda::detail::AtomicCuda const &, int * const addr, int const & value) const
            {
                return atomicSub(addr, value);
            }
        };
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Sub, unsigned int>
        {
            ALPAKA_FCT_ACC unsigned int operator()(cuda::detail::AtomicCuda const &, unsigned int * const addr, unsigned int const & value) const
            {
                return atomicSub(addr, value);
            }
        };
        //-----------------------------------------------------------------------------
        //! Min.
        //-----------------------------------------------------------------------------
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Min, int>
        {
            ALPAKA_FCT_ACC int operator()(cuda::detail::AtomicCuda const &, int * const addr, int const & value) const
            {
                return atomicMin(addr, value);
            }
        };
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Min, unsigned int>
        {
            ALPAKA_FCT_ACC unsigned int operator()(cuda::detail::AtomicCuda const &, unsigned int * const addr, unsigned int const & value) const
            {
                return atomicMin(addr, value);
            }
        };
        /*template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Min, unsigned long long int>
        {
            ALPAKA_FCT_ACC unsigned long long int operator()(cuda::detail::AtomicCuda const &, unsigned long long int * const addr, unsigned long long int const & value) const
            {
                return atomicMin(addr, value);
            }
        };*/
        //-----------------------------------------------------------------------------
        //! Max.
        //-----------------------------------------------------------------------------
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Max, int>
        {
            ALPAKA_FCT_ACC int operator()(cuda::detail::AtomicCuda const &, int * const addr, int const & value) const
            {
                return atomicMax(addr, value);
            }
        };
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Max, unsigned int>
        {
            ALPAKA_FCT_ACC unsigned int operator()(cuda::detail::AtomicCuda const &, unsigned int * const addr, unsigned int const & value) const
            {
                return atomicMax(addr, value);
            }
        };
        /*template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Max, unsigned long long int>
        {
            ALPAKA_FCT_ACC unsigned long long int operator()(cuda::detail::AtomicCuda const &, unsigned long long int * const addr, unsigned long long int const & value) const
            {
                return atomicMax(addr, value);
            }
        };*/
        //-----------------------------------------------------------------------------
        //! Exch.
        //-----------------------------------------------------------------------------
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Exch, int>
        {
            ALPAKA_FCT_ACC int operator()(cuda::detail::AtomicCuda const & , int * const addr, int const & value) const
            {
                return atomicExch(addr, value);
            }
        };
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Exch, unsigned int>
        {
            ALPAKA_FCT_ACC unsigned int operator()(cuda::detail::AtomicCuda const &, unsigned int * const addr, unsigned int const & value) const
            {
                return atomicExch(addr, value);
            }
        };
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Exch, unsigned long long int>
        {
            ALPAKA_FCT_ACC unsigned long long int operator()(cuda::detail::AtomicCuda const &, unsigned long long int * const addr, unsigned long long int const & value) const
            {
                return atomicExch(addr, value);
            }
        };
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Exch, float>
        {
            ALPAKA_FCT_ACC float operator()(cuda::detail::AtomicCuda const &, float * const addr, float const & value) const
            {
                return atomicExch(addr, value);
            }
        };
        //-----------------------------------------------------------------------------
        //! Inc.
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
        //! Dec.
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
        //! And.
        //-----------------------------------------------------------------------------
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, And, int>
        {
            ALPAKA_FCT_ACC int operator()(cuda::detail::AtomicCuda const &, int * const addr, int const & value) const
            {
                return atomicAnd(addr, value);
            }
        };
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, And, unsigned int>
        {
            ALPAKA_FCT_ACC unsigned int operator()(cuda::detail::AtomicCuda const &, unsigned int * const addr, unsigned int const & value) const
            {
                return atomicAnd(addr, value);
            }
        };
        /*template<>
        struct AtomicOp<cuda::detail::AtomicCuda, And, unsigned long long int>
        {
            ALPAKA_FCT_ACC unsigned long long int operator()(cuda::detail::AtomicCuda const &, unsigned long long int * const addr, unsigned long long int const & value) const
            {
                return atomicAnd(addr, value);
            }
        };*/
        //-----------------------------------------------------------------------------
        //! Or.
        //-----------------------------------------------------------------------------
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Or, int>
        {
            ALPAKA_FCT_ACC int operator()(cuda::detail::AtomicCuda const &, int * const addr, int const & value) const
            {
                return atomicOr(addr, value);
            }
        };
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Or, unsigned int>
        {
            ALPAKA_FCT_ACC unsigned int operator()(cuda::detail::AtomicCuda const &, unsigned int * const addr, unsigned int const & value) const
            {
                return atomicOr(addr, value);
            }
        };
        /*template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Or, unsigned long long int>
        {
            ALPAKA_FCT_ACC unsigned long long int operator()(cuda::detail::AtomicCuda const &, unsigned long long int * const addr, unsigned long long int const & value) const
            {
                return atomicOr(addr, value);
            }
        };*/
        //-----------------------------------------------------------------------------
        //! Xor.
        //-----------------------------------------------------------------------------
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Xor, int>
        {
            ALPAKA_FCT_ACC int operator()(cuda::detail::AtomicCuda const &, int * const addr, int const & value) const
            {
                return atomicXor(addr, value);
            }
        };
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Xor, unsigned int>
        {
            ALPAKA_FCT_ACC unsigned int operator()(cuda::detail::AtomicCuda const &, unsigned int * const addr, unsigned int const & value) const
            {
                return atomicXor(addr, value);
            }
        };
        /*template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Xor, unsigned long long int>
        {
            ALPAKA_FCT_ACC unsigned long long int operator()(cuda::detail::AtomicCuda const &, unsigned long long int * const addr, unsigned long long int const & value) const
            {
                return atomicXor(addr, value);
            }
        };*/
        //-----------------------------------------------------------------------------
        //! Cas.
        //-----------------------------------------------------------------------------
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Cas, int>
        {
            ALPAKA_FCT_ACC int operator()(cuda::detail::AtomicCuda const &, int * const addr, int const & compare, int const & value) const
            {
                return atomicCAS(addr, compare, value);
            }
        };
        template<>
        struct AtomicOp<cuda::detail::AtomicCuda, Cas, unsigned int>
        {
            ALPAKA_FCT_ACC unsigned int operator()(cuda::detail::AtomicCuda const &, unsigned int * const addr, unsigned int const & compare, unsigned int const & value) const
            {
                return atomicCAS(addr, compare, value);
            }
        };
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
