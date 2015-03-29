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

#include <alpaka/core/Common.hpp>   // ALPAKA_FCT_ACC

namespace alpaka
{
    namespace traits
    {
        //-----------------------------------------------------------------------------
        //! The atomic operation traits.
        //-----------------------------------------------------------------------------
        namespace atomic
        {
            //#############################################################################
            //! The abstract atomic operation functor.
            //#############################################################################
            template<
                typename TAtomic,
                typename TOp,
                typename T>
            struct AtomicOp;
        }
    }

    //-----------------------------------------------------------------------------
    //! The atomic operation traits accessors.
    //-----------------------------------------------------------------------------
    namespace atomic
    {
        //-----------------------------------------------------------------------------
        //! Executes the given operation atomically.
        //!
        //! \tparam TOp The operation type.
        //! \tparam T The value type.
        //! \tparam TAtomic The atomic implementation type.
        //! \param addr The value to change atomically.
        //! \param value The value used in the atomic operation.
        //-----------------------------------------------------------------------------
        template<
            typename TOp,
            typename T,
            typename TAtomic>
        ALPAKA_FCT_ACC auto atomicOp(
            unsigned int * const addr,
            unsigned int const & value,
            TAtomic const & atomic)
        -> T
        {
            return traits::atomic::AtomicOp<
                TAtomic, 
                TOp, 
                T>
            ::atomicOp(atomic, addr, value);
        }
    }
}
