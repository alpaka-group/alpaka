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

#include <alpaka/core/Common.hpp>   // ALPAKA_FN_HOST_ACC

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
            //! The block shared memory operation traits.
            //-----------------------------------------------------------------------------
            namespace traits
            {
                //#############################################################################
                //! The block shared variable allocation operation trait.
                //#############################################################################
                template<
                    typename T,
                    typename TBlockSharedAlloc,
                    typename TSfinae = void>
                struct AllocVar;
                //#############################################################################
                //! The block shared array allocation operation trait.
                //#############################################################################
                template<
                    typename T,
                    std::size_t TnumElements,
                    typename TBlockSharedAlloc,
                    typename TSfinae = void>
                struct AllocArr;
                //#############################################################################
                //! The block shared memory free operation trait.
                //#############################################################################
                template<
                    typename TBlockSharedAlloc,
                    typename TSfinae = void>
                struct FreeMem;
            }

            //-----------------------------------------------------------------------------
            //! Allocates a variable in block shared memory.
            //!
            //! \tparam T The element type.
            //! \tparam TBlockSharedAlloc The block shared allocator implementation type.
            //! \param blockSharedAlloc The block shared allocator implementation.
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename T,
                typename TBlockSharedAlloc>
            ALPAKA_FN_HOST_ACC auto allocVar(
                TBlockSharedAlloc const & blockSharedAlloc)
            -> T &
            {
                return traits::AllocVar<
                    T,
                    TBlockSharedAlloc>
                ::allocVar(
                    blockSharedAlloc);
            }

            //-----------------------------------------------------------------------------
            //! Allocates an array in block shared memory.
            //!
            //! \tparam T The element type.
            //! \tparam TnumElements The Number of elements.
            //! \tparam TBlockSharedAlloc The block shared allocator implementation type.
            //! \param blockSharedAlloc The block shared allocator implementation.
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename T,
                std::size_t TnumElements,
                typename TBlockSharedAlloc>
            ALPAKA_FN_HOST_ACC auto allocArr(
                TBlockSharedAlloc const & blockSharedAlloc)
            -> T *
            {
                static_assert(
                    TnumElements > 0,
                    "The number of elements to allocate in block shared memory must not be zero!");

                return traits::AllocArr<
                    T,
                    TnumElements,
                    TBlockSharedAlloc>
                ::allocArr(
                    blockSharedAlloc);
            }

            //-----------------------------------------------------------------------------
            //! Frees all block shared memory.
            //!
            //! \tparam TBlockSharedAlloc The block shared allocator implementation type.
            //! \param blockSharedAlloc The block shared allocator implementation.
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename TBlockSharedAlloc>
            ALPAKA_FN_HOST_ACC auto freeMem(
                TBlockSharedAlloc & blockSharedAlloc)
            -> void
            {
                traits::FreeMem<
                    TBlockSharedAlloc>
                ::freeMem(
                    blockSharedAlloc);
            }

            namespace traits
            {
                //#############################################################################
                //! The AllocVar trait specialization for classes with BlockSharedAllocBase member type.
                //#############################################################################
                template<
                    typename TBlockSharedAlloc,
                    typename T>
                struct AllocVar<
                    T,
                    TBlockSharedAlloc,
                    typename std::enable_if<
                        std::is_base_of<typename TBlockSharedAlloc::BlockSharedAllocBase, typename std::decay<TBlockSharedAlloc>::type>::value
                        && (!std::is_same<typename TBlockSharedAlloc::BlockSharedAllocBase, typename std::decay<TBlockSharedAlloc>::type>::value)>::type>
                {
                    //-----------------------------------------------------------------------------
                    //! \return The number of threads in each dimension of a block.
                    //-----------------------------------------------------------------------------
                    ALPAKA_NO_HOST_ACC_WARNING
                    ALPAKA_FN_HOST_ACC static auto allocVar(
                        TBlockSharedAlloc const & blockSharedAlloc)
                    -> T &
                    {
                        // Delegate the call to the base class.
                        return
                            block::shared::allocVar<
                                T>(
                                    static_cast<typename TBlockSharedAlloc::BlockSharedAllocBase const &>(blockSharedAlloc));
                    }
                };
                //#############################################################################
                //! The AllocArr trait specialization for classes with BlockSharedAllocBase member type.
                //#############################################################################
                template<
                    typename TBlockSharedAlloc,
                    typename T,
                    std::size_t TnumElements>
                struct AllocArr<
                    T,
                    TnumElements,
                    TBlockSharedAlloc,
                    typename std::enable_if<
                        std::is_base_of<typename TBlockSharedAlloc::BlockSharedAllocBase, typename std::decay<TBlockSharedAlloc>::type>::value
                        && (!std::is_same<typename TBlockSharedAlloc::BlockSharedAllocBase, typename std::decay<TBlockSharedAlloc>::type>::value)>::type>
                {
                    //-----------------------------------------------------------------------------
                    //! \return The number of threads in each dimension of a block.
                    //-----------------------------------------------------------------------------
                    ALPAKA_NO_HOST_ACC_WARNING
                    ALPAKA_FN_HOST_ACC static auto allocArr(
                        TBlockSharedAlloc const & blockSharedAlloc)
                    -> T *
                    {
                        // Delegate the call to the base class.
                        return
                            block::shared::allocArr<
                                T,
                                TnumElements>(
                                    static_cast<typename TBlockSharedAlloc::BlockSharedAllocBase const &>(blockSharedAlloc));
                    }
                };
                //#############################################################################
                //! The FreeMem trait specialization for classes with BlockSharedAllocBase member type.
                //#############################################################################
                template<
                    typename TBlockSharedAlloc>
                struct FreeMem<
                    TBlockSharedAlloc,
                    typename std::enable_if<
                        std::is_base_of<typename TBlockSharedAlloc::BlockSharedAllocBase, typename std::decay<TBlockSharedAlloc>::type>::value
                        && (!std::is_same<typename TBlockSharedAlloc::BlockSharedAllocBase, typename std::decay<TBlockSharedAlloc>::type>::value)>::type>
                {
                    //-----------------------------------------------------------------------------
                    //! \return The number of threads in each dimension of a block.
                    //-----------------------------------------------------------------------------
                    ALPAKA_NO_HOST_ACC_WARNING
                    ALPAKA_FN_HOST_ACC static auto freeMem(
                        TBlockSharedAlloc & blockSharedAlloc)
                    -> void
                    {
                        // Delegate the call to the base class.
                        block::shared::freeMem(
                            static_cast<typename TBlockSharedAlloc::BlockSharedAllocBase &>(blockSharedAlloc));
                    }
                };
            }
        }
    }
}
