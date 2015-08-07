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

#include <alpaka/atomic/Traits.hpp>                 // AtomicOp

#include <mutex>                                    // std::mutex, std::lock_guard

namespace alpaka
{
    namespace atomic
    {
        //#############################################################################
        //! The CPU threads accelerator atomic ops.
        //#############################################################################
        class AtomicStlLock
        {
        public:
            template<
                typename TAtomic,
                typename TOp,
                typename T,
                typename TSfinae>
            friend struct atomic::traits::AtomicOp;

            using AtomicBase = AtomicStlLock;

            //-----------------------------------------------------------------------------
            //! Default constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA AtomicStlLock() = default;
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA AtomicStlLock(AtomicStlLock const &) = delete;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA AtomicStlLock(AtomicStlLock &&) = delete;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA auto operator=(AtomicStlLock const &) -> AtomicStlLock & = delete;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA auto operator=(AtomicStlLock &&) -> AtomicStlLock & = delete;
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA /*virtual*/ ~AtomicStlLock() = default;

        private:
            std::mutex mutable m_mtxAtomic; //!< The mutex protecting access for a atomic operation.
        };

        namespace traits
        {
            //#############################################################################
            //! The CPU threads accelerator atomic operation function object.
            //#############################################################################
            template<
                typename TOp,
                typename T>
            struct AtomicOp<
                TOp,
                atomic::AtomicStlLock,
                T>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA static auto atomicOp(
                    atomic::AtomicStlLock const & atomic,
                    T * const addr,
                    T const & value)
                -> T
                {
                    // \TODO: Currently not only the access to the same memory location is protected by a mutex but all atomic ops on all threads.
                    // We could use a list of mutexes and lock the mutex depending on the target memory location to allow multiple atomic ops on different targets concurrently.
                    std::lock_guard<std::mutex> lock(atomic.m_mtxAtomic);
                    return TOp()(addr, value);
                }
            };
        }
    }
}
