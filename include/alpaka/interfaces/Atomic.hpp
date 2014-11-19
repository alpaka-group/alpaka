/**
* Copyright 2014 Benjamin Worpitz
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

#include <alpaka/core/Vec.hpp>          // alpaka::vec
#include <alpaka/core/Operations.hpp>   // alpaka::Add, Sub, ...

namespace alpaka
{
    namespace detail
    {
        //#############################################################################
        //! The template to execute the requested atomic operation.
        //#############################################################################
        template<typename TAtomic, typename TOp, typename T>
        struct AtomicOp;

        //#############################################################################
        //! The interface of the atomic operations.
        //#############################################################################
        template<typename TAtomic>
        class IAtomic :
            private TAtomic
        {
        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            template<typename... TArgs>
            ALPAKA_FCT_HOST_ACC IAtomic(TArgs && ... args) :
                TAtomic(std::forward<TArgs>(args)...)
            {}

            //-----------------------------------------------------------------------------
            //! Execute the atomic operation on the given address with the given value.
            //! \return The old value before executing the atomic operation.
            //-----------------------------------------------------------------------------
            template<typename TOp, typename T>
            ALPAKA_FCT_HOST_ACC T atomicOp(T * const addr, T const & value) const
            {
                return AtomicOp<TAtomic, TOp, T>()(
                    *static_cast<TAtomic const *>(this), 
                    addr,
                    value);
            }
        };
    }
}
