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

#include <alpaka/dev/Traits.hpp>
#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>

#include <alpaka/dev/cpu/SysInfo.hpp>

#include <alpaka/core/Unused.hpp>

#include <map>
#include <mutex>
#include <memory>
#include <vector>
#include <algorithm>

namespace alpaka
{
    namespace queue
    {
        class QueueCpuAsync;

        namespace cpu
        {
            namespace detail
            {
                class QueueCpuAsyncImpl;
            }
        }
    }
    namespace pltf
    {
        namespace traits
        {
            template<
                typename TPltf,
                typename TSfinae>
            struct GetDevByIdx;
        }
        class PltfCpu;
    }
    namespace dev
    {
        //-----------------------------------------------------------------------------
        //! The CPU device.
        namespace cpu
        {
            namespace detail
            {
                //#############################################################################
                //! The CPU device implementation.
                class DevCpuImpl
                {
                    friend queue::QueueCpuAsync;                   // queue::QueueCpuAsync::QueueCpuAsync calls RegisterAsyncQueue.
                    friend queue::cpu::detail::QueueCpuAsyncImpl;  // QueueCpuAsyncImpl::~QueueCpuAsyncImpl calls UnregisterAsyncQueue.
                public:
                    //-----------------------------------------------------------------------------
                    DevCpuImpl() = default;
                    //-----------------------------------------------------------------------------
                    DevCpuImpl(DevCpuImpl const &) = default;
                    //-----------------------------------------------------------------------------
                    DevCpuImpl(DevCpuImpl &&) = default;
                    //-----------------------------------------------------------------------------
                    auto operator=(DevCpuImpl const &) -> DevCpuImpl & = default;
                    //-----------------------------------------------------------------------------
                    auto operator=(DevCpuImpl &&) -> DevCpuImpl & = default;
                    //-----------------------------------------------------------------------------
                    ~DevCpuImpl() = default;

                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST auto GetAllAsyncQueueImpls() const noexcept(false)
                    -> std::vector<std::shared_ptr<queue::cpu::detail::QueueCpuAsyncImpl>>
                    {
                        std::vector<std::shared_ptr<queue::cpu::detail::QueueCpuAsyncImpl>> vspQueues;

                        std::lock_guard<std::mutex> lk(m_Mutex);

                        for(auto const & pairQueue : m_mapQueues)
                        {
                            auto spQueue(pairQueue.second.lock());
                            if(spQueue)
                            {
                                vspQueues.emplace_back(std::move(spQueue));
                            }
                            else
                            {
                                throw std::logic_error("One of the queues registered on the device is invalid!");
                            }
                        }
                        return vspQueues;
                    }

                private:
                    //-----------------------------------------------------------------------------
                    //! Registers the given queue on this device.
                    //! NOTE: Every queue has to be registered for correct functionality of device wait operations!
                    ALPAKA_FN_HOST auto RegisterAsyncQueue(std::shared_ptr<queue::cpu::detail::QueueCpuAsyncImpl> spQueueImpl)
                    -> void
                    {
                        std::lock_guard<std::mutex> lk(m_Mutex);

                        // Register this queue on the device.
                        // NOTE: We have to store the plain pointer next to the weak pointer.
                        // This is necessary to find the entry on unregistering because the weak pointer will already be invalid at that point.
                        m_mapQueues.emplace(spQueueImpl.get(), spQueueImpl);
                    }
                    //-----------------------------------------------------------------------------
                    //! Unregisters the given queue from this device.
                    ALPAKA_FN_HOST auto UnregisterAsyncQueue(queue::cpu::detail::QueueCpuAsyncImpl const * const pQueue) noexcept(false)
                    -> void
                    {
                        std::lock_guard<std::mutex> lk(m_Mutex);

                        // Unregister this queue from the device.
                        auto const itFind(std::find_if(
                            m_mapQueues.begin(),
                            m_mapQueues.end(),
                            [pQueue](std::pair<queue::cpu::detail::QueueCpuAsyncImpl *, std::weak_ptr<queue::cpu::detail::QueueCpuAsyncImpl>> const & pair)
                            {
                                return (pQueue == pair.first);
                            }));
                        if(itFind != m_mapQueues.end())
                        {
                            m_mapQueues.erase(itFind);
                        }
                        else
                        {
                            throw std::logic_error("The queue to unregister from the device could not be found in the list of registered queues!");
                        }
                    }

                private:
                    std::mutex mutable m_Mutex;
                    std::map<queue::cpu::detail::QueueCpuAsyncImpl *, std::weak_ptr<queue::cpu::detail::QueueCpuAsyncImpl>> m_mapQueues;
                };
            }
        }

        //#############################################################################
        //! The CPU device handle.
        class DevCpu
        {
            friend struct pltf::traits::GetDevByIdx<pltf::PltfCpu>;
        protected:
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST DevCpu() :
                m_spDevCpuImpl(std::make_shared<cpu::detail::DevCpuImpl>())
            {}
        public:
            //-----------------------------------------------------------------------------
            DevCpu(DevCpu const &) = default;
            //-----------------------------------------------------------------------------
            DevCpu(DevCpu &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(DevCpu const &) -> DevCpu & = default;
            //-----------------------------------------------------------------------------
            auto operator=(DevCpu &&) -> DevCpu & = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(DevCpu const &) const
            -> bool
            {
                return true;
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(DevCpu const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ~DevCpu() = default;

        public:
            std::shared_ptr<cpu::detail::DevCpuImpl> m_spDevCpuImpl;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU device name get trait specialization.
            template<>
            struct GetName<
                dev::DevCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getName(
                    dev::DevCpu const & dev)
                -> std::string
                {
                    alpaka::ignore_unused(dev);

                    return dev::cpu::detail::getCpuName();
                }
            };

            //#############################################################################
            //! The CPU device available memory get trait specialization.
            template<>
            struct GetMemBytes<
                dev::DevCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getMemBytes(
                    dev::DevCpu const & dev)
                -> std::size_t
                {
                    alpaka::ignore_unused(dev);

                    return dev::cpu::detail::getTotalGlobalMemSizeBytes();
                }
            };

            //#############################################################################
            //! The CPU device free memory get trait specialization.
            template<>
            struct GetFreeMemBytes<
                dev::DevCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getFreeMemBytes(
                    dev::DevCpu const & dev)
                -> std::size_t
                {
                    alpaka::ignore_unused(dev);

                    return dev::cpu::detail::getFreeGlobalMemSizeBytes();
                }
            };

            //#############################################################################
            //! The CPU device reset trait specialization.
            template<>
            struct Reset<
                dev::DevCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto reset(
                    dev::DevCpu const & dev)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    alpaka::ignore_unused(dev);

                    // The CPU does nothing on reset.
                }
            };
        }
    }
    namespace mem
    {
        namespace buf
        {
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            class BufCpu;

            namespace traits
            {
                //#############################################################################
                //! The CPU device memory buffer type trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct BufType<
                    dev::DevCpu,
                    TElem,
                    TDim,
                    TIdx>
                {
                    using type = mem::buf::BufCpu<TElem, TDim, TIdx>;
                };
            }
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU device platform type trait specialization.
            template<>
            struct PltfType<
                dev::DevCpu>
            {
                using type = pltf::PltfCpu;
            };
        }
    }
}
