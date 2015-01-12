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

// #define ALPAKA_OPENMP_ATOMIC_OPS_LOCK    // Undefine this to use openmp locks instead of the 'omp critical' pragma.
                                            // The pragma version should be the faster one. 
#ifdef ALPAKA_OPENMP_ATOMIC_OPS_LOCK
    #include <alpaka/openmp/Common.hpp>
#endif

#include <alpaka/interfaces/Atomic.hpp>     // IAtomic

namespace alpaka
{
    namespace openmp
    {
        namespace detail
        {
            //#############################################################################
            //! The OpenMP accelerator atomic operations.
            //#############################################################################
            class AtomicOpenMp
            {
#ifdef ALPAKA_OPENMP_ATOMIC_OPS_LOCK
            public:
                template<typename TAtomic, typename TOp, typename T>
                friend struct alpaka::detail::AtomicOp;
#endif
            public:
                //-----------------------------------------------------------------------------
                //! Default-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA AtomicOpenMp()
#ifdef ALPAKA_OPENMP_ATOMIC_OPS_LOCK
                {
                    omp_init_lock(&m_ompLock);
                }
#else
                    = default;
#endif
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA AtomicOpenMp(AtomicOpenMp const &) = default;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA AtomicOpenMp(AtomicOpenMp &&) = default;
                //-----------------------------------------------------------------------------
                //! Copy assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA AtomicOpenMp & operator=(AtomicOpenMp const &) = delete;

#ifdef ALPAKA_OPENMP_ATOMIC_OPS_LOCK
                //-----------------------------------------------------------------------------
                //! Default-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA virtual ~AtomicOpenMp()
                {
                    omp_destroy_lock(&m_ompLock);
                }
            private:
                omp_lock_t mutable m_ompLock;
#else
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA virtual ~AtomicOpenMp() noexcept = default;
#endif
            };
            using TInterfacedAtomic = alpaka::detail::IAtomic<AtomicOpenMp>;
        }
    }

    namespace detail
    {
        //#############################################################################
        //! The OpenMP accelerator atomic operation functor.
        //
        // NOTE: Can not use '#pragma omp atomic' because braces or calling other functions directly after '#pragma omp atomic' are not allowed!
        // So this would not be fully atomic! Between the store of the old value and the operation could be a context switch!
        //#############################################################################
        template<typename TOp, typename T>
        struct AtomicOp<openmp::detail::AtomicOpenMp, TOp, T>
        {
#ifdef ALPAKA_OPENMP_ATOMIC_OPS_LOCK
            ALPAKA_FCT_ACC_NO_CUDA T operator()(openmp::detail::AtomicOpenMp const & atomic, T * const addr, T const & value) const
            {
                omp_set_lock(&atomic.m_ompLock);
                auto const old(TOp()(addr, value));
                omp_unset_lock(&atomic.m_ompLock);
                return old;
            }
#else
            ALPAKA_FCT_ACC_NO_CUDA T operator()(openmp::detail::AtomicOpenMp const &, T * const addr, T const & value) const
            {
                T old;
#pragma omp critical (AtomicOp)
                {
                    old = TOp()(addr, value);
                }
                return old;
            }
#endif
        };
    }
}
