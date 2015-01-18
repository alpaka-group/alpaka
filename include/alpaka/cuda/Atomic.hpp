/**
 * \file
 * Copyright 2014-2015 Benjamin Worpitz
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

#include <alpaka/core/Common.hpp>		// ALPAKA_FCT_ACC_CUDA_ONLY

#include <alpaka/interfaces/Atomic.hpp>	// IAtomic

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
				ALPAKA_FCT_ACC_CUDA_ONLY AtomicCuda() = default;
				//-----------------------------------------------------------------------------
				//! Copy constructor.
				//-----------------------------------------------------------------------------
				ALPAKA_FCT_ACC_CUDA_ONLY AtomicCuda(AtomicCuda const &) = default;
				//-----------------------------------------------------------------------------
				//! Move constructor.
				//-----------------------------------------------------------------------------
				ALPAKA_FCT_ACC_CUDA_ONLY AtomicCuda(AtomicCuda &&) = default;
				//-----------------------------------------------------------------------------
				//! Copy assignment.
				//-----------------------------------------------------------------------------
				ALPAKA_FCT_ACC_CUDA_ONLY AtomicCuda & operator=(AtomicCuda const &) = delete;
				//-----------------------------------------------------------------------------
				//! Destructor.
				//-----------------------------------------------------------------------------
				ALPAKA_FCT_ACC_CUDA_ONLY /*virtual*/ ~AtomicCuda() noexcept = default;
			};
			using InterfacedAtomicCuda = alpaka::detail::IAtomic<AtomicCuda>;
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
		struct AtomicOp<
		    cuda::detail::AtomicCuda,
		    operations::Add,
		    int>
		{
			ALPAKA_FCT_ACC_CUDA_ONLY static int atomicOp(
				cuda::detail::AtomicCuda const &,
				int * const addr,
				int const & value)
			{
				return atomicAdd(addr, value);
			}
		};
		//-----------------------------------------------------------------------------
		//! The CUDA accelerator atomic operation functor.
		//-----------------------------------------------------------------------------
		template<>
		struct AtomicOp<
		    cuda::detail::AtomicCuda,
		    operations::Add,
		    unsigned int>
		{
			ALPAKA_FCT_ACC_CUDA_ONLY static unsigned int atomicOp(
				cuda::detail::AtomicCuda const &,
				unsigned int * const addr,
				unsigned int const & value)
			{
				return atomicAdd(addr, value);
			}
		};
		//-----------------------------------------------------------------------------
		//! The CUDA accelerator atomic operation functor.
		//-----------------------------------------------------------------------------
		template<>
		struct AtomicOp<
		    cuda::detail::AtomicCuda,
		    operations::Add,
		    unsigned long long int>
		{
			ALPAKA_FCT_ACC_CUDA_ONLY static unsigned long long int atomicOp(
				cuda::detail::AtomicCuda const &,
				unsigned long long int * const addr,
				unsigned long long int const & value)
			{
				return atomicAdd(addr, value);
			}
		};
		//-----------------------------------------------------------------------------
		//! The CUDA accelerator atomic operation functor.
		//-----------------------------------------------------------------------------
		template<>
		struct AtomicOp<
		    cuda::detail::AtomicCuda,
		    operations::Add,
		    float>
		{
			ALPAKA_FCT_ACC_CUDA_ONLY static float atomicOp(
				cuda::detail::AtomicCuda const &,
				float * const addr,
				float const & value)
			{
				return atomicAdd(addr, value);
			}
		};
		//-----------------------------------------------------------------------------
		//! The CUDA accelerator atomic operation functor.
		//-----------------------------------------------------------------------------
		template<>
		struct AtomicOp<
		    cuda::detail::AtomicCuda,
		    operations::Add,
		    double>
		{
			ALPAKA_FCT_ACC_CUDA_ONLY static double atomicOp(
				cuda::detail::AtomicCuda const &,
				double * const addr,
				double const & value)
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
				while(assumed != old);
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
		struct AtomicOp<
		    cuda::detail::AtomicCuda,
		    operations::Sub,
		    int>
		{
			ALPAKA_FCT_ACC_CUDA_ONLY static int atomicOp(
				cuda::detail::AtomicCuda const &,
				int * const addr,
				int const & value)
			{
				return atomicSub(addr, value);
			}
		};
		//-----------------------------------------------------------------------------
		//! The CUDA accelerator atomic operation functor.
		//-----------------------------------------------------------------------------
		template<>
		struct AtomicOp<
		    cuda::detail::AtomicCuda,
		    operations::Sub,
		    unsigned int>
		{
			ALPAKA_FCT_ACC_CUDA_ONLY static unsigned int atomicOp(
				cuda::detail::AtomicCuda const &,
				unsigned int * const addr,
				unsigned int const & value)
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
		struct AtomicOp<
		    cuda::detail::AtomicCuda,
		    operations::Min,
		    int>
		{
			ALPAKA_FCT_ACC_CUDA_ONLY static int atomicOp(
				cuda::detail::AtomicCuda const &,
				int * const addr,
				int const & value)
			{
				return atomicMin(addr, value);
			}
		};
		//-----------------------------------------------------------------------------
		//! The CUDA accelerator atomic operation functor.
		//-----------------------------------------------------------------------------
		template<>
		struct AtomicOp<
		    cuda::detail::AtomicCuda,
		    operations::Min,
		    unsigned int>
		{
			ALPAKA_FCT_ACC_CUDA_ONLY static unsigned int atomicOp(
				cuda::detail::AtomicCuda const &,
				unsigned int * const addr,
				unsigned int const & value)
			{
				return atomicMin(addr, value);
			}
		};
		//-----------------------------------------------------------------------------
		//! The CUDA accelerator atomic operation functor.
		//-----------------------------------------------------------------------------
        /*template<>
        struct AtomicOp<
            cuda::detail::AtomicCuda,
            operations::Min,
            unsigned long long int>
        {
            ALPAKA_FCT_ACC_CUDA_ONLY static unsigned long long int atomicOp(
                cuda::detail::AtomicCuda const &, 
                unsigned long long int * const addr, 
                unsigned long long int const & value)
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
		struct AtomicOp<
		    cuda::detail::AtomicCuda,
		    operations::Max,
		    int>
		{
			ALPAKA_FCT_ACC_CUDA_ONLY static int atomicOp(
				cuda::detail::AtomicCuda const &,
				int * const addr,
				int const & value)
			{
				return atomicMax(addr, value);
			}
		};
		//-----------------------------------------------------------------------------
		//! The CUDA accelerator atomic operation functor.
		//-----------------------------------------------------------------------------
		template<>
		struct AtomicOp<
		    cuda::detail::AtomicCuda,
		    operations::Max,
		    unsigned int>
		{
			ALPAKA_FCT_ACC_CUDA_ONLY static unsigned int atomicOp(
				cuda::detail::AtomicCuda const &,
				unsigned int * const addr,
				unsigned int const & value)
			{
				return atomicMax(addr, value);
			}
		};
		//-----------------------------------------------------------------------------
		//! The CUDA accelerator atomic operation functor.
		//-----------------------------------------------------------------------------
        /*template<>
        struct AtomicOp<
            cuda::detail::AtomicCuda,
            operations::Max,
            unsigned long long int>
        {
            ALPAKA_FCT_ACC_CUDA_ONLY static unsigned long long int atomicOp(
                cuda::detail::AtomicCuda const &, 
                unsigned long long int * const addr, 
                unsigned long long int const & value)
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
		struct AtomicOp<
		    cuda::detail::AtomicCuda,
		    operations::Exch,
		    int>
		{
			ALPAKA_FCT_ACC_CUDA_ONLY static int atomicOp(
				cuda::detail::AtomicCuda const &,
				int * const addr,
				int const & value)
			{
				return atomicExch(addr, value);
			}
		};
		//-----------------------------------------------------------------------------
		//! The CUDA accelerator atomic operation functor.
		//-----------------------------------------------------------------------------
		template<>
		struct AtomicOp<
		    cuda::detail::AtomicCuda,
		    operations::Exch,
		    unsigned int>
		{
			ALPAKA_FCT_ACC_CUDA_ONLY static unsigned int atomicOp(
				cuda::detail::AtomicCuda const &,
				unsigned int * const addr,
				unsigned int const & value)
			{
				return atomicExch(addr, value);
			}
		};
		//-----------------------------------------------------------------------------
		//! The CUDA accelerator atomic operation functor.
		//-----------------------------------------------------------------------------
		template<>
		struct AtomicOp<
		    cuda::detail::AtomicCuda,
		    operations::Exch,
		    unsigned long long int>
		{
			ALPAKA_FCT_ACC_CUDA_ONLY static unsigned long long int atomicOp(
				cuda::detail::AtomicCuda const &,
				unsigned long long int * const addr,
				unsigned long long int const & value)
			{
				return atomicExch(addr, value);
			}
		};
		//-----------------------------------------------------------------------------
		//! The CUDA accelerator atomic operation functor.
		//-----------------------------------------------------------------------------
		template<>
		struct AtomicOp<
		    cuda::detail::AtomicCuda,
		    operations::Exch,
		    float>
		{
			ALPAKA_FCT_ACC_CUDA_ONLY static float atomicOp(
				cuda::detail::AtomicCuda const &,
				float * const addr,
				float const & value)
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
		struct AtomicOp<
		    cuda::detail::AtomicCuda,
		    operations::Inc,
		    unsigned int>
		{
			ALPAKA_FCT_ACC_CUDA_ONLY static unsigned int atomicOp(
				cuda::detail::AtomicCuda const &,
				unsigned int * const addr,
				unsigned int const & value)
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
		struct AtomicOp<
		    cuda::detail::AtomicCuda,
		    operations::Dec,
		    unsigned int>
		{
			ALPAKA_FCT_ACC_CUDA_ONLY static unsigned int atomicOp(
				cuda::detail::AtomicCuda const &,
				unsigned int * const addr,
				unsigned int const & value)
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
		struct AtomicOp<
		    cuda::detail::AtomicCuda,
		    operations::And,
		    int>
		{
			ALPAKA_FCT_ACC_CUDA_ONLY static int atomicOp(
				cuda::detail::AtomicCuda const &,
				int * const addr,
				int const & value)
			{
				return atomicAnd(addr, value);
			}
		};
		//-----------------------------------------------------------------------------
		//! The CUDA accelerator atomic operation functor.
		//-----------------------------------------------------------------------------
		template<>
		struct AtomicOp<
		    cuda::detail::AtomicCuda,
		    operations::And,
		    unsigned int>
		{
			ALPAKA_FCT_ACC_CUDA_ONLY static unsigned int atomicOp(
				cuda::detail::AtomicCuda const &,
				unsigned int * const addr,
				unsigned int const & value)
			{
				return atomicAnd(addr, value);
			}
		};
		//-----------------------------------------------------------------------------
		//! The CUDA accelerator atomic operation functor.
		//-----------------------------------------------------------------------------
        /*template<>
        struct AtomicOp<
            cuda::detail::AtomicCuda,
            operations::And,
            unsigned long long int>
        {
            ALPAKA_FCT_ACC_CUDA_ONLY static unsigned long long int atomicOp(
                cuda::detail::AtomicCuda const &, 
                unsigned long long int * const addr, 
                unsigned long long int const & value)
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
		struct AtomicOp<
		cuda::detail::AtomicCuda,
		operations::Or,
		int>
		{
			ALPAKA_FCT_ACC_CUDA_ONLY static int atomicOp(
				cuda::detail::AtomicCuda const &,
				int * const addr,
				int const & value)
			{
				return atomicOr(addr, value);
			}
		};
		//-----------------------------------------------------------------------------
		//! The CUDA accelerator atomic operation functor.
		//-----------------------------------------------------------------------------
		template<>
		struct AtomicOp<
		    cuda::detail::AtomicCuda,
		    operations::Or,
		    unsigned int>
		{
			ALPAKA_FCT_ACC_CUDA_ONLY static unsigned int atomicOp(
				cuda::detail::AtomicCuda const &,
				unsigned int * const addr,
				unsigned int const & value)
			{
				return atomicOr(addr, value);
			}
		};
		//-----------------------------------------------------------------------------
		//! The CUDA accelerator atomic operation functor.
		//-----------------------------------------------------------------------------
		/*template<>
        struct AtomicOp<
            cuda::detail::AtomicCuda,
            operations::Or,
            unsigned long long int>
        {
            ALPAKA_FCT_ACC_CUDA_ONLY static unsigned long long int atomicOp(
                cuda::detail::AtomicCuda const &, 
                unsigned long long int * const addr, 
                unsigned long long int const & value)
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
		struct AtomicOp<
		    cuda::detail::AtomicCuda,
		    operations::Xor,
		    int>
		{
			ALPAKA_FCT_ACC_CUDA_ONLY static int atomicOp(
				cuda::detail::AtomicCuda const &,
				int * const addr,
				int const & value)
			{
				return atomicXor(addr, value);
			}
		};
		//-----------------------------------------------------------------------------
		//! The CUDA accelerator atomic operation functor.
		//-----------------------------------------------------------------------------
		template<>
		struct AtomicOp<
		    cuda::detail::AtomicCuda,
		    operations::Xor,
		    unsigned int>
		{
			ALPAKA_FCT_ACC_CUDA_ONLY static unsigned int atomicOp(
				cuda::detail::AtomicCuda const &,
				unsigned int * const addr,
				unsigned int const & value)
			{
				return atomicXor(addr, value);
			}
		};
		//-----------------------------------------------------------------------------
		//! The CUDA accelerator atomic operation functor.
		//-----------------------------------------------------------------------------
		/*template<>
        struct AtomicOp<
            cuda::detail::AtomicCuda,
            operations::Xor,
            unsigned long long int>
        {
            ALPAKA_FCT_ACC_CUDA_ONLY static unsigned long long int atomicOp(
                cuda::detail::AtomicCuda const &, 
                unsigned long long int * const addr, 
                unsigned long long int const & value)
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
		struct AtomicOp<
		    cuda::detail::AtomicCuda,
		    operations::Cas,
		    int>
		{
			ALPAKA_FCT_ACC_CUDA_ONLY static int atomicOp(
				cuda::detail::AtomicCuda const &,
				int * const addr,
				int const & compare,
				int const & value)
			{
				return atomicCAS(addr, compare, value);
			}
		};
		//-----------------------------------------------------------------------------
		//! The CUDA accelerator atomic operation functor.
		//-----------------------------------------------------------------------------
		template<>
		struct AtomicOp<
		    cuda::detail::AtomicCuda,
		    operations::Cas,
		    unsigned int>
		{
			ALPAKA_FCT_ACC_CUDA_ONLY static unsigned int atomicOp(
				cuda::detail::AtomicCuda const &,
				unsigned int * const addr,
				unsigned int const & compare,
				unsigned int const & value)
			{
				return atomicCAS(addr, compare, value);
			}
		};
		//-----------------------------------------------------------------------------
		//! The CUDA accelerator atomic operation functor.
		//-----------------------------------------------------------------------------
		template<>
		struct AtomicOp<
		    cuda::detail::AtomicCuda,
		    operations::Cas,
		    unsigned long long int>
		{
			ALPAKA_FCT_ACC_CUDA_ONLY static unsigned long long int atomicOp(
				cuda::detail::AtomicCuda const &,
				unsigned long long int * const addr,
				unsigned long long int const & compare,
				unsigned long long int const & value)
			{
				return atomicCAS(addr, compare, value);
			}
		};
	}
}
