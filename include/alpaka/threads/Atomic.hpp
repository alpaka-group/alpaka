/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
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

#include <mutex>                        // std::mutex, std::lock_guard

namespace alpaka
{
    namespace threads
    {
        namespace detail
        {
            //#############################################################################
            //! The threads accelerator atomic operations.
            //#############################################################################
            class AtomicThreads
            {
            public:
                template<typename TAtomic, typename TOp, typename T>
                friend struct alpaka::detail::AtomicOp;

            public:
                //-----------------------------------------------------------------------------
                //! Default-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST AtomicThreads() = default;
                //-----------------------------------------------------------------------------
                //! Copy-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST AtomicThreads(AtomicThreads const &) = default;
                //-----------------------------------------------------------------------------
                //! Move-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST AtomicThreads(AtomicThreads &&) = default;
                //-----------------------------------------------------------------------------
                //! Copy-assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST AtomicThreads & operator=(AtomicThreads const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST ~AtomicThreads() noexcept = default;

            private:
                std::mutex mutable m_mtxAtomic; //!< The mutex protecting access for a atomic operation.
            };
            using TInterfacedAtomic = alpaka::detail::IAtomic<AtomicThreads>;
        }
    }

    namespace detail
    {
        //#############################################################################
        //! The threads accelerator atomic operation functor.
        //#############################################################################
        template<typename TOp, typename T>
        struct AtomicOp<threads::detail::AtomicThreads, TOp, T>
        {
            ALPAKA_FCT_HOST T operator()(threads::detail::AtomicThreads const & atomic, T * const addr, T const & value) const
            {
                // \TODO: Currently not only the access to the same memory location is protected by a mutex but all atomic ops on all threads.
                // We could use a list of mutexes and lock the mutex depending on the target memory location to allow multiple atomic ops on different targets concurrently.
                std::lock_guard<std::mutex> lock(atomic.m_mtxAtomic);
                return TOp()(addr, value);
            }
        };
    }
}
