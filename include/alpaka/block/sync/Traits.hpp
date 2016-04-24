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

#include <alpaka/meta/IsStrictBase.hpp> // meta::IsStrictBase

#include <alpaka/core/Common.hpp>       // ALPAKA_FN_HOST_ACC

#include <type_traits>                  // std::enable_if

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The grid block specifics
    //-----------------------------------------------------------------------------
    namespace block
    {
        //-----------------------------------------------------------------------------
        //! The block synchronization specifics.
        //-----------------------------------------------------------------------------
        namespace sync
        {
            //-----------------------------------------------------------------------------
            //! The block synchronization traits.
            //-----------------------------------------------------------------------------
            namespace traits
            {
                //#############################################################################
                //! The block synchronization operation trait.
                //#############################################################################
                template<
                    typename TBlockSync,
                    typename TSfinae = void>
                struct SyncBlockThread;
            }

            //-----------------------------------------------------------------------------
            //! Allocates a variable in block shared memory.
            //!
            //! \tparam TBlockSync The block synchronization implementation type.
            //! \param blockSync The block synchronization implementation.
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename TBlockSync>
            ALPAKA_FN_ACC auto syncBlockThreads(
                TBlockSync const & blockSync)
            -> void
            {
                traits::SyncBlockThread<
                    TBlockSync>
                ::syncBlockThreads(
                    blockSync);
            }

            namespace traits
            {
                //#############################################################################
                //! The AllocVar trait specialization for classes with BlockSyncBase member type.
                //#############################################################################
                template<
                    typename TBlockSync>
                struct SyncBlockThread<
                    TBlockSync,
                    typename std::enable_if<
                        meta::IsStrictBase<
                            typename TBlockSync::BlockSyncBase,
                            TBlockSync
                        >::value
                    >::type>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_NO_HOST_ACC_WARNING
                    ALPAKA_FN_ACC static auto syncBlockThreads(
                        TBlockSync const & blockSync)
                    -> void
                    {
                        // Delegate the call to the base class.
                        block::sync::syncBlockThreads(
                                static_cast<typename TBlockSync::BlockSyncBase const &>(blockSync));
                    }
                };
            }
        }
    }
}
