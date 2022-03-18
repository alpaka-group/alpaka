/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, Jan Stephan, Bernhard Manfred Gruber,
 * Antonio Di Pilato
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/dev/Traits.hpp>
#include <alpaka/dev/cpu/SysInfo.hpp>
#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/queue/Properties.hpp>
#include <alpaka/queue/QueueGenericThreadsBlocking.hpp>
#include <alpaka/queue/QueueGenericThreadsNonBlocking.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/queue/cpu/IGenericThreadsQueue.hpp>
#include <alpaka/traits/Traits.hpp>
#include <alpaka/wait/Traits.hpp>

#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace alpaka
{
    class DevCpu;
    namespace cpu
    {
        using ICpuQueue = IGenericThreadsQueue<DevCpu>;
    }
    namespace trait
    {
        template<typename TPltf, typename TSfinae>
        struct GetDevByIdx;
    }
    class PltfCpu;

    //! The CPU device.
    namespace cpu::detail
    {
        //! The CPU device implementation.
        class DevCpuImpl
        {
        public:
            ALPAKA_FN_HOST auto getAllExistingQueues() const -> std::vector<std::shared_ptr<cpu::ICpuQueue>>
            {
                std::vector<std::shared_ptr<cpu::ICpuQueue>> vspQueues;

                std::lock_guard<std::mutex> lk(m_Mutex);
                vspQueues.reserve(std::size(m_queues));

                for(auto it = std::begin(m_queues); it != std::end(m_queues);)
                {
                    auto spQueue(it->lock());
                    if(spQueue)
                    {
                        vspQueues.emplace_back(std::move(spQueue));
                        ++it;
                    }
                    else
                    {
                        it = m_queues.erase(it);
                    }
                }
                return vspQueues;
            }

            //! Registers the given queue on this device.
            //! NOTE: Every queue has to be registered for correct functionality of device wait operations!
            ALPAKA_FN_HOST auto registerQueue(std::shared_ptr<cpu::ICpuQueue> spQueue) const -> void
            {
                std::lock_guard<std::mutex> lk(m_Mutex);

                // Register this queue on the device.
                m_queues.push_back(spQueue);
            }

        private:
            std::mutex mutable m_Mutex;
            std::vector<std::weak_ptr<cpu::ICpuQueue>> mutable m_queues;
        };
    } // namespace cpu::detail

    //! The CPU device handle.
    class DevCpu
        : public concepts::Implements<ConceptCurrentThreadWaitFor, DevCpu>
        , public concepts::Implements<ConceptDev, DevCpu>
    {
        friend struct trait::GetDevByIdx<PltfCpu>;

    protected:
        DevCpu() : m_spDevCpuImpl(std::make_shared<cpu::detail::DevCpuImpl>())
        {
        }

    public:
        auto operator==(DevCpu const&) const -> bool
        {
            return true;
        }
        auto operator!=(DevCpu const& rhs) const -> bool
        {
            return !((*this) == rhs);
        }

        [[nodiscard]] ALPAKA_FN_HOST auto getAllQueues() const -> std::vector<std::shared_ptr<cpu::ICpuQueue>>
        {
            return m_spDevCpuImpl->getAllExistingQueues();
        }

        //! Registers the given queue on this device.
        //! NOTE: Every queue has to be registered for correct functionality of device wait operations!
        ALPAKA_FN_HOST auto registerQueue(std::shared_ptr<cpu::ICpuQueue> spQueue) const -> void
        {
            m_spDevCpuImpl->registerQueue(spQueue);
        }

        [[nodiscard]] auto getNativeHandle() const noexcept
        {
            return 0;
        }

    private:
        std::shared_ptr<cpu::detail::DevCpuImpl> m_spDevCpuImpl;
    };

    namespace trait
    {
        //! The CPU device name get trait specialization.
        template<>
        struct GetName<DevCpu>
        {
            ALPAKA_FN_HOST static auto getName(DevCpu const& /* dev */) -> std::string
            {
                return cpu::detail::getCpuName();
            }
        };

        //! The CPU device available memory get trait specialization.
        template<>
        struct GetMemBytes<DevCpu>
        {
            ALPAKA_FN_HOST static auto getMemBytes(DevCpu const& /* dev */) -> std::size_t
            {
                return cpu::detail::getTotalGlobalMemSizeBytes();
            }
        };

        //! The CPU device free memory get trait specialization.
        template<>
        struct GetFreeMemBytes<DevCpu>
        {
            ALPAKA_FN_HOST static auto getFreeMemBytes(DevCpu const& /* dev */) -> std::size_t
            {
                return cpu::detail::getFreeGlobalMemSizeBytes();
            }
        };

        //! The CPU device warp size get trait specialization.
        template<>
        struct GetWarpSizes<DevCpu>
        {
            ALPAKA_FN_HOST static auto getWarpSizes(DevCpu const& /* dev */) -> std::vector<std::size_t>
            {
                return {1u};
            }
        };

        //! The CPU device reset trait specialization.
        template<>
        struct Reset<DevCpu>
        {
            ALPAKA_FN_HOST static auto reset(DevCpu const& /* dev */) -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;
                // The CPU does nothing on reset.
            }
        };

        //! The CPU device native handle type trait specialization.
        template<>
        struct NativeHandle<DevCpu>
        {
            [[nodiscard]] static auto getNativeHandle(DevCpu const& dev)
            {
                return dev.getNativeHandle();
            }
        };
    } // namespace trait

    template<typename TElem, typename TDim, typename TIdx>
    class BufCpu;

    namespace trait
    {
        //! The CPU device memory buffer type trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct BufType<DevCpu, TElem, TDim, TIdx>
        {
            using type = BufCpu<TElem, TDim, TIdx>;
        };

        //! The CPU device platform type trait specialization.
        template<>
        struct PltfType<DevCpu>
        {
            using type = PltfCpu;
        };
    } // namespace trait
    using QueueCpuNonBlocking = QueueGenericThreadsNonBlocking<DevCpu>;
    using QueueCpuBlocking = QueueGenericThreadsBlocking<DevCpu>;

    namespace trait
    {
        template<>
        struct QueueType<DevCpu, Blocking>
        {
            using type = QueueCpuBlocking;
        };

        template<>
        struct QueueType<DevCpu, NonBlocking>
        {
            using type = QueueCpuNonBlocking;
        };
    } // namespace trait
} // namespace alpaka
