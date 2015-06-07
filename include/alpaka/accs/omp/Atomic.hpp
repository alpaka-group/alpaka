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

#include <alpaka/traits/Atomic.hpp>     // AtomicOp

#include <boost/core/ignore_unused.hpp> // boost::ignore_unused

namespace alpaka
{
    namespace accs
    {
        namespace omp
        {
            namespace detail
            {
                //#############################################################################
                //! The OpenMP accelerator atomic ops.
                //#############################################################################
                class AtomicOmp
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Default constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA AtomicOmp() = default;
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA AtomicOmp(AtomicOmp const &) = delete;
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA AtomicOmp(AtomicOmp &&) = delete;
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA auto operator=(AtomicOmp const &) -> AtomicOmp & = delete;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA auto operator=(AtomicOmp &&) -> AtomicOmp & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA /*virtual*/ ~AtomicOmp() = default;
                };
            }
        }
    }

    namespace traits
    {
        namespace atomic
        {
            //#############################################################################
            //! The OpenMP accelerator atomic operation functor.
            //
            // NOTE: Can not use '#pragma omp atomic' because braces or calling other functions directly after '#pragma omp atomic' are not allowed!
            // So this would not be fully atomic! Between the store of the old value and the operation could be a context switch!
            //#############################################################################
            template<
                typename TOp,
                typename T>
            struct AtomicOp<
                accs::omp::detail::AtomicOmp,
                TOp,
                T>
            {
                ALPAKA_FCT_ACC_NO_CUDA static auto atomicOp(
                    accs::omp::detail::AtomicOmp const & atomic,
                    T * const addr,
                    T const & value)
                -> T
                {
                    boost::ignore_unused(atomic);
                    T old;
                    #pragma omp critical (AlpakaOmpAtomicOp)
                    {
                        old = TOp()(addr, value);
                    }
                    return old;
                }
            };
        }
    }
}
