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

#include <alpaka/core/Vec.hpp>          // Vec

#include <boost/core/ignore_unused.hpp> // boost::ignore_unused

//-----------------------------------------------------------------------------
//! The name space for the accelerator library.
//-----------------------------------------------------------------------------
namespace alpaka
{
    //#############################################################################
    //! The trait for getting the size of the block shared extern memory of a kernel.
    //!
    //! The default implementation returns 0.
    //#############################################################################
    template<
        typename TAccelereatedKernel, 
        typename TSfinae = void>
    struct BlockSharedExternMemSizeBytes
    {
        //-----------------------------------------------------------------------------
        //! \param v3uiBlockThreadExtents The size of the blocks for which the block shared memory size should be calculated.
        //! \tparam TArgs The kernel invocation argument types pack.
        //! \param ... The kernel invocation arguments for which the block shared memory size should be calculated.
        //! \return The size of the shared memory allocated for a block in bytes.
        //! The default version always returns zero.
        //-----------------------------------------------------------------------------
        template<
            typename TAcc,
            typename... TArgs>
        ALPAKA_FCT_HOST static UInt getBlockSharedExternMemSizeBytes(
            Vec<3u> const & v3uiBlockThreadExtents, 
            TArgs && ... )
        {
            boost::ignore_unused(v3uiBlockThreadExtents);

            return 0;
        }
    };
}
