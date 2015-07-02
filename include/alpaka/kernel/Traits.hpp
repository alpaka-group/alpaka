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
#include <alpaka/core/Common.hpp>       // ALPAKA_FCT_HOST_ACC

#include <boost/core/ignore_unused.hpp> // boost::ignore_unused

//-----------------------------------------------------------------------------
//! The alpaka accelerator library.
//-----------------------------------------------------------------------------
namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The kernel specifics.
    //-----------------------------------------------------------------------------
    namespace kernel
    {
        //-----------------------------------------------------------------------------
        //! The kernel traits.
        //-----------------------------------------------------------------------------
        namespace traits
        {
            //#############################################################################
            //! The trait for getting the size of the block shared extern memory of a kernel.
            //!
            //! \tparam TKernelFctObj The kernel function object.
            //! \tparam TAcc The accelerator.
            //!
            //! The default implementation returns 0.
            //#############################################################################
            template<
                typename TKernelFctObj,
                typename TAcc,
                typename TSfinae = void>
            struct BlockSharedExternMemSizeBytes
            {
                //-----------------------------------------------------------------------------
                //! \param vuiBlockThreadExtents The size of the blocks for which the block shared memory size should be calculated.
                //! \tparam TArgs The kernel invocation argument types pack.
                //! \param args,... The kernel invocation arguments for which the block shared memory size should be calculated.
                //! \return The size of the shared memory allocated for a block in bytes.
                //! The default version always returns zero.
                //-----------------------------------------------------------------------------
                template<
                    typename TDim,
                    typename... TArgs>
                ALPAKA_FCT_HOST_ACC static auto getBlockSharedExternMemSizeBytes(
                    Vec<TDim, size::SizeT<TAcc>> const & vuiBlockThreadExtents,
                    TArgs const & ... args)
                -> size::SizeT<TAcc>
                {
                    boost::ignore_unused(vuiBlockThreadExtents);
                    boost::ignore_unused(args...);

                    return 0;
                }
            };
        }

        //-----------------------------------------------------------------------------
        //! \param vuiBlockThreadExtents The size of the blocks for which the block shared memory size should be calculated.
        //! \tparam TArgs The kernel invocation argument types pack.
        //! \param args,... The kernel invocation arguments for which the block shared memory size should be calculated.
        //! \return The size of the shared memory allocated for a block in bytes.
        //! The default version always returns zero.
        //-----------------------------------------------------------------------------
        template<
            typename TKernelFctObj,
            typename TAcc,
            typename TDim,
            typename... TArgs>
        ALPAKA_FCT_HOST_ACC auto getBlockSharedExternMemSizeBytes(
            Vec<TDim, size::SizeT<TAcc>> const & vuiBlockThreadExtents,
            TArgs const & ... args)
        -> size::SizeT<TAcc>
        {
            return traits::BlockSharedExternMemSizeBytes<
                TKernelFctObj,
                TAcc>
            ::getBlockSharedExternMemSizeBytes(
                vuiBlockThreadExtents,
                args...);
        }
    }
}
