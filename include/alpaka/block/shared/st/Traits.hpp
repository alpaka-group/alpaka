/**
* \file
* Copyright 2014-2015 Benjamin Worpitz, Rene Widera
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

#include <alpaka/core/Common.hpp>   // ALPAKA_FN_HOST_ACC

#include <alpaka/meta/UniqueId.hpp> // meta::uniqueId

#include <type_traits>              // std::enable_if, std::is_base_of, std::is_same, std::decay

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The grid block specifics
    //-----------------------------------------------------------------------------
    namespace block
    {
        //-----------------------------------------------------------------------------
        //! The block shared memory operation specifics.
        //-----------------------------------------------------------------------------
        namespace shared
        {
            //-----------------------------------------------------------------------------
            //! The block shared static memory operation specifics.
            //-----------------------------------------------------------------------------
            namespace st
            {
                //-----------------------------------------------------------------------------
                //! The block shared static memory operation traits.
                //-----------------------------------------------------------------------------
                namespace traits
                {
                    //#############################################################################
                    //! The block shared static memory variable allocation operation trait.
                    //#############################################################################
                    template<
                        typename T,
                        std::size_t TuniqueId,
                        typename TBlockSharedMemSt,
                        typename TSfinae = void>
                    struct AllocVar;
                    //#############################################################################
                    //! The block shared static memory array allocation operation trait.
                    //#############################################################################
                    template<
                        typename T,
                        std::size_t TnumElements,
                        std::size_t TuniqueId,
                        typename TBlockSharedMemSt,
                        typename TSfinae = void>
                    struct AllocArr;
                    //#############################################################################
                    //! The block shared static memory free operation trait.
                    //#############################################################################
                    template<
                        typename TBlockSharedMemSt,
                        typename TSfinae = void>
                    struct FreeMem;
                }

                //-----------------------------------------------------------------------------
                //! Allocates a variable in block shared static memory.
                //!
                //! \tparam T The element type.
                //! \tparam TuniqueId id those is unique inside a kernel
                //! \tparam TBlockSharedMemSt The block shared allocator implementation type.
                //! \param blockSharedMemSt The block shared allocator implementation.
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename T,
                    typename TBlockSharedMemSt,
                    std::size_t TuniqueId = meta::uniqueId()>
                ALPAKA_FN_HOST_ACC auto allocVar(
                    TBlockSharedMemSt const & blockSharedMemSt)
                -> T &
                {
                    return
                        traits::AllocVar<
                            T,
                            TuniqueId,
                            TBlockSharedMemSt>
                        ::allocVar(
                            blockSharedMemSt);
                }

                //-----------------------------------------------------------------------------
                //! Allocates an array in block shared static memory.
                //!
                //! \tparam T The element type.
                //! \tparam TnumElements The Number of elements.
                //! \tparam TuniqueId id those is unique inside a kernel
                //! \tparam TBlockSharedMemSt The block shared allocator implementation type.
                //! \param blockSharedMemSt The block shared allocator implementation.
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename T,
                    std::size_t TnumElements,
                    typename TBlockSharedMemSt,
                    std::size_t TuniqueId = meta::uniqueId()>
                ALPAKA_FN_HOST_ACC auto allocArr(
                    TBlockSharedMemSt const & blockSharedMemSt)
                -> T *
                {
                    static_assert(
                        TnumElements > 0,
                        "The number of elements to allocate in block shared memory must not be zero!");

                    return
                        traits::AllocArr<
                            T,
                            TnumElements,
                            TuniqueId,
                            TBlockSharedMemSt>
                        ::allocArr(
                            blockSharedMemSt);
                }

                //-----------------------------------------------------------------------------
                //! Frees all block shared static memory.
                //!
                //! \tparam TBlockSharedMemSt The block shared allocator implementation type.
                //! \param blockSharedMemSt The block shared allocator implementation.
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TBlockSharedMemSt>
                ALPAKA_FN_HOST_ACC auto freeMem(
                    TBlockSharedMemSt & blockSharedMemSt)
                -> void
                {
                    traits::FreeMem<
                        TBlockSharedMemSt>
                    ::freeMem(
                        blockSharedMemSt);
                }

                namespace traits
                {
                    //#############################################################################
                    //! The AllocVar trait specialization for classes with BlockSharedMemStBase member type.
                    //#############################################################################
                    template<
                        typename T,
                        std::size_t TuniqueId,
                        typename TBlockSharedMemSt>
                    struct AllocVar<
                        T,
                        TuniqueId,
                        TBlockSharedMemSt,
                        typename std::enable_if<
                            std::is_base_of<typename TBlockSharedMemSt::BlockSharedMemStBase, typename std::decay<TBlockSharedMemSt>::type>::value
                            && (!std::is_same<typename TBlockSharedMemSt::BlockSharedMemStBase, typename std::decay<TBlockSharedMemSt>::type>::value)>::type>
                    {
                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        ALPAKA_NO_HOST_ACC_WARNING
                        ALPAKA_FN_HOST_ACC static auto allocVar(
                            TBlockSharedMemSt const & blockSharedMemSt)
                        -> T &
                        {
                            // Delegate the call to the base class.
                            return
                                block::shared::st::allocVar<
                                    T,
                                    typename TBlockSharedMemSt::BlockSharedMemStBase,
                                    TuniqueId>(
                                        static_cast<typename TBlockSharedMemSt::BlockSharedMemStBase const &>(blockSharedMemSt));
                        }
                    };
                    //#############################################################################
                    //! The AllocArr trait specialization for classes with BlockSharedMemStBase member type.
                    //#############################################################################
                    template<
                        typename T,
                        std::size_t TnumElements,
                        std::size_t TuniqueId,
                        typename TBlockSharedMemSt>
                    struct AllocArr<
                        T,
                        TnumElements,
                        TuniqueId,
                        TBlockSharedMemSt,
                        typename std::enable_if<
                            std::is_base_of<typename TBlockSharedMemSt::BlockSharedMemStBase, typename std::decay<TBlockSharedMemSt>::type>::value
                            && (!std::is_same<typename TBlockSharedMemSt::BlockSharedMemStBase, typename std::decay<TBlockSharedMemSt>::type>::value)>::type>
                    {
                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        ALPAKA_NO_HOST_ACC_WARNING
                        ALPAKA_FN_HOST_ACC static auto allocArr(
                            TBlockSharedMemSt const & blockSharedMemSt)
                        -> T *
                        {
                            // Delegate the call to the base class.
                            return
                                block::shared::st::allocArr<
                                    T,
                                    TnumElements,
                                    typename TBlockSharedMemSt::BlockSharedMemStBase,
                                    TuniqueId>(
                                        static_cast<typename TBlockSharedMemSt::BlockSharedMemStBase const &>(blockSharedMemSt));
                        }
                    };
                    //#############################################################################
                    //! The FreeMem trait specialization for classes with BlockSharedMemStBase member type.
                    //#############################################################################
                    template<
                        typename TBlockSharedMemSt>
                    struct FreeMem<
                        TBlockSharedMemSt,
                        typename std::enable_if<
                            std::is_base_of<typename TBlockSharedMemSt::BlockSharedMemStBase, typename std::decay<TBlockSharedMemSt>::type>::value
                            && (!std::is_same<typename TBlockSharedMemSt::BlockSharedMemStBase, typename std::decay<TBlockSharedMemSt>::type>::value)>::type>
                    {
                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        ALPAKA_NO_HOST_ACC_WARNING
                        ALPAKA_FN_HOST_ACC static auto freeMem(
                            TBlockSharedMemSt & blockSharedMemSt)
                        -> void
                        {
                            // Delegate the call to the base class.
                            block::shared::st::freeMem(
                                static_cast<typename TBlockSharedMemSt::BlockSharedMemStBase &>(blockSharedMemSt));
                        }
                    };
                }
            }
        }
    }
}
