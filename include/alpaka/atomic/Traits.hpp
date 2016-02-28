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

#include <alpaka/core/Common.hpp>       // ALPAKA_FN_HOST_ACC

#include <alpaka/meta/IsStrictBase.hpp> // meta::IsStrictBase

#include <type_traits>                  // std::enable_if

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The atomic operation traits specifics.
    //-----------------------------------------------------------------------------
    namespace atomic
    {
        //-----------------------------------------------------------------------------
        //! The atomic operation traits.
        //-----------------------------------------------------------------------------
        namespace traits
        {
            //#############################################################################
            //! The atomic operation trait.
            //#############################################################################
            template<
                typename TOp,
                typename TAtomic,
                typename T,
                typename TSfinae = void>
            struct AtomicOp;
        }

        //-----------------------------------------------------------------------------
        //! Executes the given operation atomically.
        //!
        //! \tparam TOp The operation type.
        //! \tparam T The value type.
        //! \tparam TAtomic The atomic implementation type.
        //! \param addr The value to change atomically.
        //! \param value The value used in the atomic operation.
        //! \param atomic The atomic implementation.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TOp,
            typename TAtomic,
            typename T>
        ALPAKA_FN_HOST_ACC auto atomicOp(
            TAtomic const & atomic,
            T * const addr,
            T const & value)
        -> T
        {
            return
                traits::AtomicOp<
                    TOp,
                    TAtomic,
                    T>
                ::atomicOp(
                    atomic,
                    addr,
                    value);
        }

        //-----------------------------------------------------------------------------
        //! Executes the given operation atomically.
        //!
        //! \tparam TOp The operation type.
        //! \tparam TAtomic The atomic implementation type.
        //! \tparam T The value type.
        //! \param atomic The atomic implementation.
        //! \param addr The value to change atomically.
        //! \param compare The comparison value used in the atomic operation.
        //! \param value The value used in the atomic operation.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TOp,
            typename TAtomic,
            typename T>
        ALPAKA_FN_HOST_ACC auto atomicOp(
            TAtomic const & atomic,
            T * const addr,
            T const & compare,
            T const & value)
        -> T
        {
            return
                traits::AtomicOp<
                    TOp,
                    TAtomic,
                    T>
                ::atomicOp(
                    atomic,
                    addr,
                    compare,
                    value);
        }

        namespace traits
        {
            //#############################################################################
            //! The AtomicOp trait specialization for classes with AtomicBase member type.
            //#############################################################################
            template<
                typename TOp,
                typename TAtomic,
                typename T>
            struct AtomicOp<
                TOp,
                TAtomic,
                T,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename TAtomic::AtomicBase,
                        TAtomic
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto atomicOp(
                    TAtomic const & atomic,
                    T * const addr,
                    T const & value)
                -> T
                {
                    // Delegate the call to the base class.
                    return
                        atomic::atomicOp<
                            TOp>(
                                static_cast<typename TAtomic::AtomicBase const &>(atomic),
                                addr,
                                value);
                }
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto atomicOp(
                    TAtomic const & atomic,
                    T * const addr,
                    T const & compare,
                    T const & value)
                -> T
                {
                    // Delegate the call to the base class.
                    return
                        atomic::atomicOp<
                            TOp>(
                                static_cast<typename TAtomic::AtomicBase const &>(atomic),
                                addr,
                                compare,
                                value);
                }
            };
        }
    }
}
