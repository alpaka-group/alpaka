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

// #define ALPAKA_OPENMP_ATOMIC_OPS_LOCK    // Undefine this to use omp2 locks instead of the 'omp2 critical' pragma.
                                            // The pragma version should be the faster one.
#ifdef ALPAKA_OPENMP_ATOMIC_OPS_LOCK
    #include <alpaka/accs/omp/Common.hpp>
#endif

#include <alpaka/traits/Atomic.hpp>         // AtomicOp

namespace alpaka
{
    namespace accs
    {
        namespace omp
        {
            namespace omp2
            {
                namespace detail
                {
                    //#############################################################################
                    //! The OpenMP2 accelerator atomic ops.
                    //#############################################################################
                    class AtomicOmp2
                    {
#ifdef ALPAKA_OPENMP_ATOMIC_OPS_LOCK
                    public:
                        template<
                            typename TAtomic,
                            typename TOp,
                            typename T>
                        friend struct alpaka::traits::atomic::AtomicOp;
#endif
                    public:
                        //-----------------------------------------------------------------------------
                        //! Default constructor.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_ACC_NO_CUDA AtomicOmp2()
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
                        ALPAKA_FCT_ACC_NO_CUDA AtomicOmp2(AtomicOmp2 const &) = default;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                        //-----------------------------------------------------------------------------
                        //! Move constructor.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_ACC_NO_CUDA AtomicOmp2(AtomicOmp2 &&) = default;
#endif
                        //-----------------------------------------------------------------------------
                        //! Copy assignment.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_ACC_NO_CUDA auto operator=(AtomicOmp2 const &) -> AtomicOmp2 & = delete;

#ifdef ALPAKA_OPENMP_ATOMIC_OPS_LOCK
                        //-----------------------------------------------------------------------------
                        //! Default constructor.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_ACC_NO_CUDA virtual ~AtomicOmp2()
                        {
                            omp_destroy_lock(&m_ompLock);
                        }
                    private:
                        omp_lock_t mutable m_ompLock;
#else
                        //-----------------------------------------------------------------------------
                        //! Destructor.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_ACC_NO_CUDA virtual ~AtomicOmp2() noexcept = default;
#endif
                    };
                }
            }
        }
    }

    namespace traits
    {
        namespace atomic
        {
            //#############################################################################
            //! The OpenMP2 accelerator atomic operation functor.
            //
            // NOTE: Can not use '#pragma omp atomic' because braces or calling other functions directly after '#pragma omp atomic' are not allowed!
            // So this would not be fully atomic! Between the store of the old value and the operation could be a context switch!
            //#############################################################################
            template<
                typename TOp,
                typename T>
            struct AtomicOp<
                accs::omp::omp2::detail::AtomicOmp2,
                TOp,
                T>
            {
#ifdef ALPAKA_OPENMP_ATOMIC_OPS_LOCK
                ALPAKA_FCT_ACC_NO_CUDA static auto atomicOp(
                    accs::omp::omp2::detail::AtomicOmp2 const & atomic,
                    T * const addr,
                    T const & value)
                -> T
                {
                    omp_set_lock(&atomic.m_ompLock);
                    auto const old(TOp()(addr, value));
                    omp_unset_lock(&atomic.m_ompLock);
                    return old;
                }
#else
                ALPAKA_FCT_ACC_NO_CUDA static auto atomicOp(
                    accs::omp::omp2::detail::AtomicOmp2 const &,
                    T * const addr,
                    T const & value)
                -> T
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
}
