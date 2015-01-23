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

#include <alpaka/interfaces/Atomic.hpp> // IAtomic

namespace alpaka
{
    namespace serial
    {
        namespace detail
        {
            //#############################################################################
            //! The serial accelerator atomic operations.
            //#############################################################################
            class AtomicSerial
            {
            public:
                //-----------------------------------------------------------------------------
                //! Default constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA AtomicSerial() = default;
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA AtomicSerial(AtomicSerial const &) = default;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA AtomicSerial(AtomicSerial &&) = default;
                //-----------------------------------------------------------------------------
                //! Copy assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA AtomicSerial & operator=(AtomicSerial const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA virtual ~AtomicSerial() noexcept = default;
            };
            using InterfacedAtomicSerial = alpaka::detail::IAtomic<AtomicSerial>;
        }
    }

    namespace detail
    {
        //#############################################################################
        //! The serial accelerator atomic operation functor.
        //#############################################################################
        template<
            typename TOp, 
            typename T>
        struct AtomicOp<
            serial::detail::AtomicSerial, 
            TOp, 
            T>
        {
            ALPAKA_FCT_ACC_NO_CUDA static T atomicOp(
                serial::detail::AtomicSerial const &, 
                T * const addr, 
                T const & value)
            {
                return TOp()(addr, value);
            }
        };
    }
}
