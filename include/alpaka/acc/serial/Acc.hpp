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
#include <alpaka/workdiv/WorkDivMembers.hpp>    // workdiv::WorkDivMembers
#include <alpaka/idx/gb/IdxGbRef.hpp>           // IdxGbRef
#include <alpaka/idx/bt/IdxBtZero.hpp>          // IdxBtZero
#include <alpaka/atomic/AtomicNoOp.hpp>         // AtomicNoOp

// Specialized traits.
#include <alpaka/acc/Traits.hpp>                // AccType
#include <alpaka/dev/Traits.hpp>                // DevType
#include <alpaka/exec/Traits.hpp>               // ExecType

// Implementation details.
#include <alpaka/dev/DevCpu.hpp>                // DevCpu

#include <boost/core/ignore_unused.hpp>         // boost::ignore_unused
#include <boost/align.hpp>                      // boost::aligned_alloc

#include <memory>                               // std::unique_ptr
#include <vector>                               // std::vector

namespace alpaka
{
    namespace exec
    {
        template<
            typename TDim>
        class ExecCpuSerial;

        namespace serial
        {
            namespace detail
            {
                template<
                    typename TDim>
                class ExecCpuSerialImpl;
            }
        }
    }
    namespace acc
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
                //#############################################################################
                //! The CPU serial accelerator.
                //!
                //! This accelerator allows serial kernel execution on a CPU device.
                //! The block size is restricted to 1x1x1 and all blocks are executed serially so there is no parallelism at all.
                //#############################################################################
                template<
                    typename TDim>
                class AccCpuSerial final :
                    public workdiv::WorkDivMembers<TDim>,
                    public idx::gb::IdxGbRef<TDim>,
                    public idx::bt::IdxBtZero<TDim>,
                    public atomic::AtomicNoOp
                {
                public:
                    friend class ::alpaka::exec::serial::detail::ExecCpuSerialImpl<TDim>;

                private:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TWorkDiv>
                    ALPAKA_FCT_ACC_NO_CUDA AccCpuSerial(
                        TWorkDiv const & workDiv) :
                            workdiv::WorkDivMembers<TDim>(workDiv),
                            idx::gb::IdxGbRef<TDim>(m_vuiGridBlockIdx),
                            idx::bt::IdxBtZero<TDim>(),
                            atomic::AtomicNoOp(),
                            m_vuiGridBlockIdx(Vec<TDim>::zeros())
                    {}

                public:
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    // Do not copy most members because they are initialized by the executor for each execution.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA AccCpuSerial(AccCpuSerial const &) = delete;
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA AccCpuSerial(AccCpuSerial &&) = delete;
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA auto operator=(AccCpuSerial const &) -> AccCpuSerial & = delete;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA auto operator=(AccCpuSerial &&) -> AccCpuSerial & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA /*virtual*/ ~AccCpuSerial() = default;

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

                        m_vvuiSharedMem.emplace_back(
                            reinterpret_cast<uint8_t *>(
                                boost::alignment::aligned_alloc(16u, sizeof(T) * TuiNumElements)));
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
                    alignas(16u) Vec<TDim> mutable m_vuiGridBlockIdx;            //!< The index of the currently executed block.

                    // allocBlockSharedMem
                    std::vector<
                        std::unique_ptr<uint8_t, boost::alignment::aligned_delete>> mutable m_vvuiSharedMem;    //!< Block shared memory.

                    // getBlockSharedExternMem
                    std::unique_ptr<uint8_t, boost::alignment::aligned_delete> mutable m_vuiExternalSharedMem;  //!< External block shared memory.
                };
            }
        }
    }

    template<
        typename TDim>
    using AccCpuSerial = acc::serial::detail::AccCpuSerial<TDim>;

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU serial accelerator accelerator type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct AccType<
                acc::serial::detail::AccCpuSerial<TDim>>
            {
                using type = acc::serial::detail::AccCpuSerial<TDim>;
            };
            //#############################################################################
            //! The CPU serial accelerator device properties get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetAccDevProps<
                acc::serial::detail::AccCpuSerial<TDim>>
            {
                ALPAKA_FCT_HOST static auto getAccDevProps(
                    dev::DevCpu const & dev)
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
                acc::serial::detail::AccCpuSerial<TDim>>
            {
                ALPAKA_FCT_HOST_ACC static auto getAccName()
                -> std::string
                {
                    return "AccCpuSerial<" + std::to_string(TDim::value) + ">";
                }
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU serial accelerator device type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DevType<
                acc::serial::detail::AccCpuSerial<TDim>>
            {
                using type = dev::DevCpu;
            };
            //#############################################################################
            //! The CPU serial accelerator device type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DevManType<
                acc::serial::detail::AccCpuSerial<TDim>>
            {
                using type = dev::DevManCpu;
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU serial accelerator dimension getter trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DimType<
                acc::serial::detail::AccCpuSerial<TDim>>
            {
                using type = TDim;
            };
        }
    }
    namespace exec
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU serial accelerator executor type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct ExecType<
                acc::serial::detail::AccCpuSerial<TDim>>
            {
                using type = exec::ExecCpuSerial<TDim>;
            };
        }
    }
}
