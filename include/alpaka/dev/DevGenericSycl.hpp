/* Copyright 2020 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#include <alpaka/core/Common.hpp>

#if !BOOST_LANG_SYCL
    #error If ALPAKA_ACC_SYCL_ENABLED is set, the compiler has to support SYCL!
#endif

#include <alpaka/acc/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/queue/Properties.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/wait/Traits.hpp>
#include <alpaka/core/Sycl.hpp>

#include <CL/sycl.hpp>

#include <shared_mutex>
#include <vector>

namespace alpaka
{
    template <typename TElem, typename TDim, typename TIdx, typename TDev>
    class BufGenericSycl;

    template <typename TDev>
    class QueueGenericSyclBlocking;

    template <typename TDev>
    class QueueGenericSyclNonBlocking;

    //#############################################################################
    //! The SYCL device handle.
    template <typename TPltf>
    class DevGenericSycl : public concepts::Implements<ConceptCurrentThreadWaitFor, DevGenericSycl<TPltf>>
                         , public concepts::Implements<ConceptDev, DevGenericSycl<TPltf>>
    {
        friend struct traits::GetDevByIdx<TPltf>;
        friend struct traits::GetName<DevGenericSycl<TPltf>>;
        friend struct traits::GetMemBytes<DevGenericSycl<TPltf>>;
        friend struct traits::GetWarpSize<DevGenericSycl<TPltf>>;
        
        template<typename TElem, typename TIdx, typename TDim, typename TDev, typename TSfinae>
        friend struct traits::BufAlloc;

        template<typename TAcc, typename TSfinae>
        friend struct traits::GetAccDevProps;

        template<typename TDev, typename TTask, typename TSfinae>
        friend struct traits::Enqueue;

        template<typename TElem, typename TIdx, typename TDim, typename TDev>
        friend class BufGenericSycl;
        friend class QueueGenericSyclBlocking<DevGenericSycl<TPltf>>;
        friend class QueueGenericSyclNonBlocking<DevGenericSycl<TPltf>>;

    public:
        DevGenericSycl(cl::sycl::device device, cl::sycl::context context)
        : m_device{device}, m_context{context}
        {}

        //-----------------------------------------------------------------------------
        DevGenericSycl(DevGenericSycl const &) = default;
        //-----------------------------------------------------------------------------
        DevGenericSycl(DevGenericSycl &&) = default;
        //-----------------------------------------------------------------------------
        auto operator=(DevGenericSycl const &) -> DevGenericSycl & = default;
        //-----------------------------------------------------------------------------
        auto operator=(DevGenericSycl &&) -> DevGenericSycl & = default;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator==(DevGenericSycl const & rhs) const -> bool
        {
            return (rhs.m_device == m_device);
        }
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator!=(DevGenericSycl const & rhs) const -> bool
        {
            return !operator==(rhs);
        }
        //-----------------------------------------------------------------------------
        ~DevGenericSycl() = default;

    private:
        DevGenericSycl() = default;

        cl::sycl::device m_device;
        cl::sycl::context m_context;
        std::vector<cl::sycl::event> m_dependencies = {};
        std::shared_ptr<std::shared_mutex> mutable mutex_ptr{std::make_shared<std::shared_mutex>()};
    };

    namespace traits
    {
        //#############################################################################
        //! The SYCL device name get trait specialization.
        template<typename TPltf>
        struct GetName<DevGenericSycl<TPltf>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getName(DevGenericSycl<TPltf> const& dev) -> std::string
            {
                return dev.m_device.template get_info<cl::sycl::info::device::name>();
            }
        };

        //#############################################################################
        //! The SYCL device available memory get trait specialization.
        template<typename TPltf>
        struct GetMemBytes<DevGenericSycl<TPltf>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getMemBytes(DevGenericSycl<TPltf> const& dev) -> std::size_t
            {
                return dev.m_device.template get_info<cl::sycl::info::device::global_mem_size>();
            }
        };

        //#############################################################################
        //! The SYCL device free memory get trait specialization. Note that
        //! this function will usually return the size of the device memory
        //! as there is no standard way in SYCL to query free memory.
        template<typename TPltf>
        struct GetFreeMemBytes<DevGenericSycl<TPltf>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getFreeMemBytes(DevGenericSycl<TPltf> const& dev) -> std::size_t
            {
                // There is no way in SYCL to query free memory. If you find a way be sure to update the
                // documentation above.
                std::cerr << "[SYCL] Warning: Querying free device memory not supported for SYCL devices.\n";
                return getMemBytes(dev);
            }
        };

        //#############################################################################
        //! The SYCL device warp size get trait specialization.
        template<typename TPltf>
        struct GetWarpSize<
            DevGenericSycl<TPltf>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getWarpSize(
                DevGenericSycl<TPltf> const & dev)
            -> std::size_t
            {
                const auto sizes = dev.m_device.template get_info<cl::sycl::info::device::sub_group_sizes>();
                return *(std::min_element(std::begin(sizes), std::end(sizes)));
            }
        };

        //#############################################################################
        //! The SYCL device reset trait specialization. Note that this
        //! function won't actually do anything. If you need to reset your
        //! SYCL device its destructor must be called.
        template<typename TPltf>
        struct Reset<DevGenericSycl<TPltf>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto reset(DevGenericSycl<TPltf> const&) -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;
                std::cerr << "[SYCL] Warning: Explicit device reset not supported for SYCL devices\n";
            }
        };

        //#############################################################################
        //! The SYCL device memory buffer type trait specialization.
        template<typename TElem, typename TDim, typename TIdx, typename TPltf>
        struct BufType<DevGenericSycl<TPltf>, TElem, TDim, TIdx>
        {
            using type = BufGenericSycl<TElem, TDim, TIdx, DevGenericSycl<TPltf>>;
        };

        //#############################################################################
        //! The SYCL device platform type trait specialization.
        template<typename TPltf>
        struct PltfType<DevGenericSycl<TPltf>>
        {
            using type = TPltf;
        };

        //#############################################################################
        //! The thread SYCL device wait specialization.
        //!
        //! Blocks until the device has completed all preceding requested tasks.
        //! Tasks that are enqueued or queues that are created after this call is made are not waited for.
        template<typename TPltf>
        struct CurrentThreadWaitFor<DevGenericSycl<TPltf>>
        {
            ALPAKA_FN_HOST static auto currentThreadWaitFor(DevGenericSycl<TPltf> const&) -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;
                std::cerr << "[SYCL] Warning: You cannot wait for SYCL devices. Use the queue instead.\n";
            }
        };

	//#############################################################################
	//! The SYCL blocking queue trait specialization.
        template<typename TPltf>
        struct QueueType<DevGenericSycl<TPltf>, Blocking>
        {
            using type = QueueGenericSyclBlocking<DevGenericSycl<TPltf>>;
        };

	//#############################################################################
	//! The SYCL non-blocking queue trait specialization.
        template<typename TPltf>
        struct QueueType<DevGenericSycl<TPltf>, NonBlocking>
        {
            using type = QueueGenericSyclNonBlocking<TPltf>;
        };
    }
}

#endif
