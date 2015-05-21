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

// Specialized traits.
#include <alpaka/traits/Acc.hpp>            // AccType
#include <alpaka/traits/Exec.hpp>           // ExecType
#include <alpaka/traits/Event.hpp>          // EventType
#include <alpaka/traits/Dev.hpp>            // DevType
#include <alpaka/traits/Stream.hpp>         // StreamType

// Implementation details.
#include <alpaka/accs/serial/Acc.hpp>       // AccCpuSerial
#include <alpaka/core/BasicWorkDiv.hpp>     // workdiv::BasicWorkDiv
#include <alpaka/core/NdLoop.hpp>           // NdLoop
#include <alpaka/devs/cpu/Dev.hpp>          // DevCpu
#include <alpaka/devs/cpu/Event.hpp>        // EventCpu
#include <alpaka/devs/cpu/Stream.hpp>       // StreamCpu
#include <alpaka/traits/Kernel.hpp>         // BlockSharedExternMemSizeBytes

#include <boost/core/ignore_unused.hpp>     // boost::ignore_unused

#include <cassert>                          // assert
#include <utility>                          // std::forward
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <iostream>                     // std::cout
#endif

namespace alpaka
{
    namespace accs
    {
        namespace serial
        {
            namespace detail
            {
                //#############################################################################
                //! The CPU serial executor.
                //#############################################################################
                template<
                    typename TDim>
                class ExecCpuSerial :
                    private AccCpuSerial<TDim>
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TWorkDiv>
                    ALPAKA_FCT_HOST ExecCpuSerial(
                        TWorkDiv const & workDiv,
                        devs::cpu::detail::StreamCpu & stream) :
                            AccCpuSerial<TDim>(workDiv),
                            m_Stream(stream)
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    }
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST ExecCpuSerial(
                        ExecCpuSerial const & other) :
                            AccCpuSerial<TDim>(static_cast<alpaka::workdiv::BasicWorkDiv<TDim> const &>(other)),
                            m_Stream(other.m_Stream)
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    }
    #if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST ExecCpuSerial(
                        ExecCpuSerial && other) :
                            AccCpuSerial<TDim>(static_cast<alpaka::workdiv::BasicWorkDiv<TDim> &&>(other)),
                            m_Stream(other.m_Stream)
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    }
    #endif
                    //-----------------------------------------------------------------------------
                    //! Copy assignment.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator=(ExecCpuSerial const &) -> ExecCpuSerial & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
    #if BOOST_COMP_INTEL
                    ALPAKA_FCT_HOST virtual ~ExecCpuSerial() = default;
    #else
                    ALPAKA_FCT_HOST virtual ~ExecCpuSerial() noexcept = default;
    #endif

                    //-----------------------------------------------------------------------------
                    //! Executes the kernel functor.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TKernelFunctor,
                        typename... TArgs>
                    ALPAKA_FCT_HOST auto operator()(
                        TKernelFunctor && kernelFunctor,
                        TArgs && ... args) const
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        auto const vuiGridBlockExtents(this->AccCpuSerial<TDim>::template getWorkDiv<Grid, Blocks>());
                        auto const vuiBlockThreadExtents(this->AccCpuSerial<TDim>::template getWorkDiv<Block, Threads>());

                        auto const uiBlockSharedExternMemSizeBytes(
                            kernel::getBlockSharedExternMemSizeBytes<
                                typename std::decay<TKernelFunctor>::type,
                                AccCpuSerial<TDim>>(
                                    vuiBlockThreadExtents,
                                    std::forward<TArgs>(args)...));
    #if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        std::cout << BOOST_CURRENT_FUNCTION
                            << " BlockSharedExternMemSizeBytes: " << uiBlockSharedExternMemSizeBytes << " B"
                            << std::endl;
    #endif
                        if(uiBlockSharedExternMemSizeBytes > 0)
                        {
                            this->AccCpuSerial<TDim>::m_vuiExternalSharedMem.reset(
                                new uint8_t[uiBlockSharedExternMemSizeBytes]);
                        }

                        // Execute the blocks serially.
                        ndLoop(
                            vuiGridBlockExtents,
                            [&](Vec<TDim> const & vuiBlockThreadIdx)
                            {
                                this->AccCpuSerial<TDim>::m_vuiGridBlockIdx = vuiBlockThreadIdx;
                                assert(vuiBlockThreadExtents.prod() == 1u);

                                // There is only ever one thread in a block in the serial accelerator.
                                std::forward<TKernelFunctor>(kernelFunctor)(
                                    (*static_cast<AccCpuSerial<TDim> const *>(this)),
                                    std::forward<TArgs>(args)...);

                                // After a block has been processed, the shared memory has to be deleted.
                                this->AccCpuSerial<TDim>::m_vvuiSharedMem.clear();
                            });

                        // After all blocks have been processed, the external shared memory has to be deleted.
                        this->AccCpuSerial<TDim>::m_vuiExternalSharedMem.reset();
                    }

                public:
                    devs::cpu::detail::StreamCpu m_Stream;
                };
            }
        }
    }

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The CPU serial executor accelerator type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct AccType<
                accs::serial::detail::ExecCpuSerial<TDim>>
            {
                using type = accs::serial::detail::AccCpuSerial<TDim>;
            };
        }

        namespace dev
        {
            //#############################################################################
            //! The CPU serial executor device type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DevType<
                accs::serial::detail::ExecCpuSerial<TDim>>
            {
                using type = devs::cpu::detail::DevCpu;
            };
            //#############################################################################
            //! The CPU serial executor device manager type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DevManType<
                accs::serial::detail::ExecCpuSerial<TDim>>
            {
                using type = devs::cpu::detail::DevManCpu;
            };
        }

        namespace dim
        {
            //#############################################################################
            //! The CPU serial executor dimension getter trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DimType<
                accs::serial::detail::ExecCpuSerial<TDim>>
            {
                using type = TDim;
            };
        }

        namespace event
        {
            //#############################################################################
            //! The CPU serial executor event type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct EventType<
                accs::serial::detail::ExecCpuSerial<TDim>>
            {
                using type = devs::cpu::detail::EventCpu;
            };
        }

        namespace exec
        {
            //#############################################################################
            //! The CPU serial executor executor type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct ExecType<
                accs::serial::detail::ExecCpuSerial<TDim>>
            {
                using type = accs::serial::detail::ExecCpuSerial<TDim>;
            };
        }

        namespace stream
        {
            //#############################################################################
            //! The CPU serial executor stream type trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct StreamType<
                accs::serial::detail::ExecCpuSerial<TDim>>
            {
                using type = devs::cpu::detail::StreamCpu;
            };
            //#############################################################################
            //! The CPU serial executor stream get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetStream<
                accs::serial::detail::ExecCpuSerial<TDim>>
            {
                ALPAKA_FCT_HOST static auto getStream(
                    accs::serial::detail::ExecCpuSerial<TDim> const & exec)
                -> devs::cpu::detail::StreamCpu
                {
                    return exec.m_Stream;
                }
            };
        }
    }
}
