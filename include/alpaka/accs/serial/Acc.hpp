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

// Base classes.
#include <alpaka/core/BasicWorkDiv.hpp>     // workdiv::BasicWorkDiv
#include <alpaka/accs/serial/Idx.hpp>       // IdxSerial
#include <alpaka/accs/serial/Atomic.hpp>    // AtomicSerial

// Specialized traits.
#include <alpaka/traits/Acc.hpp>            // AccType
#include <alpaka/traits/Exec.hpp>           // ExecType
#include <alpaka/traits/Dev.hpp>            // DevType

// Implementation details.
#include <alpaka/devs/cpu/Dev.hpp>          // DevCpu

#include <boost/core/ignore_unused.hpp>     // boost::ignore_unused

#include <memory>                           // std::unique_ptr
#include <vector>                           // std::vector

namespace alpaka
{
    namespace accs
    {
        //-----------------------------------------------------------------------------
        //! The CPU serial accelerator.
        //-----------------------------------------------------------------------------
        namespace serial
        {
            //-----------------------------------------------------------------------------
            //! The CPU serial accelerator implementation details.
            //-----------------------------------------------------------------------------
            namespace detail
            {
                // Forward declaration.
                template<
                    typename TDim>
                class ExecCpuSerial;

                //#############################################################################
                //! The CPU serial accelerator.
                //!
                //! This accelerator allows serial kernel execution on a CPU device.
                //! The block size is restricted to 1x1x1 so there is no parallelism at all.
                //#############################################################################
                template<
                    typename TDim>
                class AccCpuSerial :
                    protected alpaka::workdiv::BasicWorkDiv<TDim>,
                    protected IdxSerial<TDim>,
                    protected AtomicSerial
                {
                public:
                    friend class ::alpaka::accs::serial::detail::ExecCpuSerial<TDim>;

                private:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TWorkDiv>
                    ALPAKA_FCT_ACC_NO_CUDA AccCpuSerial(
                        TWorkDiv const & workDiv) :
                            alpaka::workdiv::BasicWorkDiv<TDim>(workDiv),
                            IdxSerial<TDim>(m_vuiGridBlockIdx),
                            AtomicSerial(),
                            m_vuiGridBlockIdx(Vec<TDim>::zeros())
                    {}

                public:
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    // Do not copy most members because they are initialized by the executor for each execution.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA AccCpuSerial(AccCpuSerial const &) = delete;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA AccCpuSerial(AccCpuSerial &&) = delete;
#endif
                    //-----------------------------------------------------------------------------
                    //! Copy assignment.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA auto operator=(AccCpuSerial const &) -> AccCpuSerial & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA virtual ~AccCpuSerial() noexcept = default;

                    //-----------------------------------------------------------------------------
                    //! \return The requested indices.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TOrigin,
                        typename TUnit>
                    ALPAKA_FCT_ACC_NO_CUDA auto getIdx() const
                    -> Vec<TDim>
                    {
                        return idx::getIdx<TOrigin, TUnit>(
                            *static_cast<IdxSerial<TDim> const *>(this),
                            *static_cast<alpaka::workdiv::BasicWorkDiv<TDim> const *>(this));
                    }

                    //-----------------------------------------------------------------------------
                    //! \return The requested extents.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TOrigin,
                        typename TUnit>
                    ALPAKA_FCT_ACC_NO_CUDA auto getWorkDiv() const
                    -> Vec<TDim>
                    {
                        return workdiv::getWorkDiv<TOrigin, TUnit>(
                            *static_cast<alpaka::workdiv::BasicWorkDiv<TDim> const *>(this));
                    }

                    //-----------------------------------------------------------------------------
                    //! Execute the atomic operation on the given address with the given value.
                    //! \return The old value before executing the atomic operation.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TOp,
                        typename T>
                    ALPAKA_FCT_ACC auto atomicOp(
                        T * const addr,
                        T const & value) const
                    -> T
                    {
                        return atomic::atomicOp<TOp, T>(
                            addr,
                            value,
                            *static_cast<AtomicSerial const *>(this));
                    }

                    //-----------------------------------------------------------------------------
                    //! Syncs all threads in the current block.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA void syncBlockThreads() const
                    {
                        // Nothing to do in here because only one thread in a group is allowed.
                    }

                    //-----------------------------------------------------------------------------
                    //! \return Allocates block shared memory.
                    //-----------------------------------------------------------------------------
                    template<
                        typename T,
                        UInt TuiNumElements>
                    ALPAKA_FCT_ACC_NO_CUDA auto allocBlockSharedMem() const
                    -> T *
                    {
                        static_assert(TuiNumElements > 0, "The number of elements to allocate in block shared memory must not be zero!");

                        // \TODO: C++14 std::make_unique would be better.
                        m_vvuiSharedMem.emplace_back(
                            std::unique_ptr<uint8_t[]>(
                                reinterpret_cast<uint8_t*>(new T[TuiNumElements])));
                        return reinterpret_cast<T*>(m_vvuiSharedMem.back().get());
                    }

                    //-----------------------------------------------------------------------------
                    //! \return The pointer to the externally allocated block shared memory.
                    //-----------------------------------------------------------------------------
                    template<
                        typename T>
                    ALPAKA_FCT_ACC_NO_CUDA auto getBlockSharedExternMem() const
                    -> T *
                    {
                        return reinterpret_cast<T*>(m_vuiExternalSharedMem.get());
                    }

                private:
                    // getIdx
                    Vec<TDim> mutable m_vuiGridBlockIdx;                        //!< The index of the currently executed block.

                    // allocBlockSharedMem
                    std::vector<
                        std::unique_ptr<uint8_t[]>> mutable m_vvuiSharedMem;    //!< Block shared memory.

                    // getBlockSharedExternMem
                    std::unique_ptr<uint8_t[]> mutable m_vuiExternalSharedMem;  //!< External block shared memory.
                };
            }
        }
    }

    template<
        typename TDim>
    using AccCpuSerial = accs::serial::detail::AccCpuSerial<TDim>;

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The CPU serial accelerator accelerator type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct AccType<
                accs::serial::detail::AccCpuSerial<TDim>>
            {
                using type = accs::serial::detail::AccCpuSerial<TDim>;
            };
            //#############################################################################
            //! The CPU serial accelerator device properties get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetAccDevProps<
                accs::serial::detail::AccCpuSerial<TDim>>
            {
                ALPAKA_FCT_HOST static auto getAccDevProps(
                    devs::cpu::DevCpu const & dev)
                -> alpaka::acc::AccDevProps<TDim>
                {
                    boost::ignore_unused(dev);

                    return {
                        // m_uiMultiProcessorCount
                        1u,
                        // m_uiBlockThreadsCountMax
                        1u,
                        // m_vuiBlockThreadExtentsMax
                        Vec<TDim>::ones(),
                        // m_vuiGridBlockExtentsMax
                        Vec<TDim>::all(std::numeric_limits<typename Vec<TDim>::Val>::max())};
                }
            };
            //#############################################################################
            //! The CPU serial accelerator name trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetAccName<
                accs::serial::detail::AccCpuSerial<TDim>>
            {
                ALPAKA_FCT_HOST_ACC static auto getAccName()
                -> std::string
                {
                    return "AccCpuSerial<" + std::to_string(TDim::value) + ">";
                }
            };
        }

        namespace dev
        {
            //#############################################################################
            //! The CPU serial accelerator device type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DevType<
                accs::serial::detail::AccCpuSerial<TDim>>
            {
                using type = devs::cpu::DevCpu;
            };
            //#############################################################################
            //! The CPU serial accelerator device type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DevManType<
                accs::serial::detail::AccCpuSerial<TDim>>
            {
                using type = devs::cpu::DevManCpu;
            };
        }

        namespace dim
        {
            //#############################################################################
            //! The CPU serial accelerator dimension getter trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DimType<
                accs::serial::detail::AccCpuSerial<TDim>>
            {
                using type = TDim;
            };
        }

        namespace exec
        {
            //#############################################################################
            //! The CPU serial accelerator executor type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct ExecType<
                accs::serial::detail::AccCpuSerial<TDim>>
            {
                using type = accs::serial::detail::ExecCpuSerial<TDim>;
            };
        }
    }
}
