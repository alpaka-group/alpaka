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
#include <alpaka/accs/serial/Acc.hpp>       // AccSerial
#include <alpaka/core/BasicWorkDiv.hpp>     // workdiv::BasicWorkDiv
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
                //! The serial accelerator executor.
                //#############################################################################
                class ExecSerial :
                    private AccSerial
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TWorkDiv>
                    ALPAKA_FCT_HOST ExecSerial(
                        TWorkDiv const & workDiv,
                        devs::cpu::detail::StreamCpu & stream) :
                            AccSerial(workDiv),
                            m_Stream(stream)
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    }
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST ExecSerial(
                        ExecSerial const & other) :
                            AccSerial(static_cast<alpaka::workdiv::BasicWorkDiv const &>(other)),
                            m_Stream(other.m_Stream)
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    }
    #if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST ExecSerial(
                        ExecSerial && other) :
                            AccSerial(static_cast<alpaka::workdiv::BasicWorkDiv &&>(other)),
                            m_Stream(other.m_Stream)
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    }
    #endif
                    //-----------------------------------------------------------------------------
                    //! Copy assignment.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator=(ExecSerial const &) -> ExecSerial & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
    #if BOOST_COMP_INTEL
                    ALPAKA_FCT_HOST virtual ~ExecSerial() = default;
    #else
                    ALPAKA_FCT_HOST virtual ~ExecSerial() noexcept = default;
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

                        Vec3<> const v3uiGridBlockExtents(this->AccSerial::getWorkDiv<Grid, Blocks, dim::Dim3>());
                        Vec3<> const v3uiBlockThreadExtents(this->AccSerial::getWorkDiv<Block, Threads, dim::Dim3>());

                        auto const uiBlockSharedExternMemSizeBytes(kernel::getBlockSharedExternMemSizeBytes<typename std::decay<TKernelFunctor>::type, AccSerial>(
                            v3uiBlockThreadExtents,
                            std::forward<TArgs>(args)...));
    #if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        std::cout << BOOST_CURRENT_FUNCTION
                            << " BlockSharedExternMemSizeBytes: " << uiBlockSharedExternMemSizeBytes << " B"
                            << std::endl;
    #endif
                        this->AccSerial::m_vuiExternalSharedMem.reset(
                            new uint8_t[uiBlockSharedExternMemSizeBytes]);

                        // Execute the blocks serially.
                        for(this->AccSerial::m_v3uiGridBlockIdx[2] = 0; this->AccSerial::m_v3uiGridBlockIdx[2]<v3uiGridBlockExtents[2]; ++this->AccSerial::m_v3uiGridBlockIdx[2])
                        {
                            for(this->AccSerial::m_v3uiGridBlockIdx[1] = 0; this->AccSerial::m_v3uiGridBlockIdx[1]<v3uiGridBlockExtents[1]; ++this->AccSerial::m_v3uiGridBlockIdx[1])
                            {
                                for(this->AccSerial::m_v3uiGridBlockIdx[0] = 0; this->AccSerial::m_v3uiGridBlockIdx[0]<v3uiGridBlockExtents[0]; ++this->AccSerial::m_v3uiGridBlockIdx[0])
                                {
                                    assert(v3uiBlockThreadExtents[0] == 1);
                                    assert(v3uiBlockThreadExtents[1] == 1);
                                    assert(v3uiBlockThreadExtents[2] == 1);

                                    // There is only ever one thread in a block in the serial accelerator.
                                    std::forward<TKernelFunctor>(kernelFunctor)(
                                        (*static_cast<AccSerial const *>(this)),
                                        std::forward<TArgs>(args)...);

                                    // After a block has been processed, the shared memory can be deleted.
                                    this->AccSerial::m_vvuiSharedMem.clear();
                                }
                            }
                        }
                        // After all blocks have been processed, the external shared memory can be deleted.
                        this->AccSerial::m_vuiExternalSharedMem.reset();
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
            //! The serial accelerator executor accelerator type trait specialization.
            //#############################################################################
            template<>
            struct AccType<
                accs::serial::detail::ExecSerial>
            {
                using type = accs::serial::detail::AccSerial;
            };
        }

        namespace event
        {
            //#############################################################################
            //! The serial accelerator executor event type trait specialization.
            //#############################################################################
            template<>
            struct EventType<
                accs::serial::detail::ExecSerial>
            {
                using type = devs::cpu::detail::EventCpu;
            };
        }

        namespace exec
        {
            //#############################################################################
            //! The serial accelerator executor executor type trait specialization.
            //#############################################################################
            template<>
            struct ExecType<
                accs::serial::detail::ExecSerial>
            {
                using type = accs::serial::detail::ExecSerial;
            };
        }

        namespace dev
        {
            //#############################################################################
            //! The serial accelerator executor device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                accs::serial::detail::ExecSerial>
            {
                using type = devs::cpu::detail::DevCpu;
            };
            //#############################################################################
            //! The serial accelerator device type trait specialization.
            //#############################################################################
            template<>
            struct DevManType<
                accs::serial::detail::ExecSerial>
            {
                using type = devs::cpu::detail::DevManCpu;
            };
        }

        namespace stream
        {
            //#############################################################################
            //! The serial accelerator executor stream type trait specialization.
            //#############################################################################
            template<>
            struct StreamType<
                accs::serial::detail::ExecSerial>
            {
                using type = devs::cpu::detail::StreamCpu;
            };
            //#############################################################################
            //! The serial accelerator executor stream get trait specialization.
            //#############################################################################
            template<>
            struct GetStream<
                accs::serial::detail::ExecSerial>
            {
                ALPAKA_FCT_HOST static auto getStream(
                    accs::serial::detail::ExecSerial const & exec)
                -> devs::cpu::detail::StreamCpu
                {
                    return exec.m_Stream;
                }
            };
        }
    }
}
