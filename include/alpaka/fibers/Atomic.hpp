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

namespace alpaka
{
    namespace fibers
    {
        namespace detail
        {
            //#############################################################################
            //! The fibers accelerator atomic operations.
            //#############################################################################
            class AtomicFibers
            {
            public:
                //-----------------------------------------------------------------------------
                //! Default-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST AtomicFibers() = default;
                //-----------------------------------------------------------------------------
                //! Copy-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST AtomicFibers(AtomicFibers const &) = default;
                //-----------------------------------------------------------------------------
                //! Move-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST AtomicFibers(AtomicFibers &&) = default;
                //-----------------------------------------------------------------------------
                //! Copy-assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST AtomicFibers & operator=(AtomicFibers const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST ~AtomicFibers() noexcept = default;
            };
            using TInterfacedAtomic = alpaka::detail::IAtomic<AtomicFibers>;
        }
    }

    namespace detail
    {
        //#############################################################################
        //! The fibers accelerator atomic operation functor.
        //#############################################################################
        template<typename TOp, typename T>
        struct AtomicOp<fibers::detail::AtomicFibers, TOp, T>
        {
            ALPAKA_FCT_HOST T operator()(fibers::detail::AtomicFibers const &, T * const addr, T const & value) const
            {
                return TOp()(addr, value);
            }
        };
    }
}
