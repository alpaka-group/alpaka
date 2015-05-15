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
        namespace serial
        {
            namespace detail
            {
                //#############################################################################
                //! The serial accelerator atomic ops.
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
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA AtomicSerial(AtomicSerial &&) = default;
#endif
                    //-----------------------------------------------------------------------------
                    //! Copy assignment.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA auto operator=(AtomicSerial const &) -> AtomicSerial & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA virtual ~AtomicSerial() noexcept = default;
                };
            }
        }
    }


    namespace traits
    {
        namespace atomic
        {
            //#############################################################################
            //! The serial accelerator atomic operation functor.
            //#############################################################################
            template<
                typename TOp,
                typename T>
            struct AtomicOp<
                accs::serial::detail::AtomicSerial,
                TOp,
                T>
            {
                ALPAKA_FCT_ACC_NO_CUDA static auto atomicOp(
                    accs::serial::detail::AtomicSerial const & atomic,
                    T * const addr,
                    T const & value)
                -> T
                {
                    boost::ignore_unused(atomic);
                    return TOp()(addr, value);
                }
            };
        }
    }
}
