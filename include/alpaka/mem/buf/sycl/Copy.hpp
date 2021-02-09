/* Copyright 2021 Jan Stephan
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
#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/dev/DevGenericSycl.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/elem/Traits.hpp>
#include <alpaka/extent/Traits.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/core/Sycl.hpp>

#include <CL/sycl.hpp>

#include <memory>
#include <shared_mutex>
#include <type_traits>
#include <vector>

namespace alpaka
{
    namespace detail
    {
        template <typename TElem>
        struct TaskCopySyclImpl
        {
            TaskCopySyclImpl(TElem const* src, TElem* dst, std::size_t size)
            : src_ptr{src}, dst_ptr{dst}, bytes{size}
            {}

            TaskCopySyclImpl(TaskCopySyclImpl const&) = delete;
            auto operator=(TaskCopySyclImpl const&) -> TaskCopySyclImpl& = delete;
            TaskCopySyclImpl(TaskCopySyclImpl&&) = default;
            auto operator=(TaskCopySyclImpl&&) -> TaskCopySyclImpl& = default;
            ~TaskCopySyclImpl() = default;

            TElem const* src_ptr;
            TElem* dst_ptr;
            std::size_t bytes;
            std::vector<cl::sycl::event> dependencies = {};
            std::shared_mutex mutex{};
        };

        //#############################################################################
        //! The SYCL memory copy trait.
        template <typename TElem>
        struct TaskCopySycl
        {
            auto operator()(cl::sycl::handler& cgh) -> void
            {
                cgh.depends_on(pimpl->dependencies);
                cgh.memcpy(pimpl->dst_ptr, pimpl->src_ptr, pimpl->bytes);
            }

            std::shared_ptr<TaskCopySyclImpl<TElem>> pimpl;
            // Distinguish from non-alpaka types (= host tasks)
            static constexpr auto is_sycl_enqueueable = true;
        };
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for CreateTaskMemcpy.
    namespace traits
    {
        //#############################################################################
        //! The SYCL host-to-device memory copy trait specialization.
        template<typename TDim, typename TPltf>
        struct CreateTaskMemcpy<TDim, DevGenericSycl<TPltf>, DevCpu>
        {
            //-----------------------------------------------------------------------------
            template<typename TExtent, typename TViewSrc, typename TViewDst>
            ALPAKA_FN_HOST static auto createTaskMemcpy(TViewDst & viewDst, TViewSrc const& viewSrc, TExtent const & ext)
            {
                using SrcType = Elem<std::remove_const_t<TViewSrc>>;
                constexpr auto SrcBytes = sizeof(SrcType);

                static_assert(!std::is_const<TViewDst>::value, "The destination view cannot be const!");

                static_assert(Dim<TViewDst>::value == Dim<std::remove_const_t<TViewSrc>>::value,
                              "The source and the destination view are required to have the same dimensionality!");

                static_assert(Dim<TViewDst>::value == Dim<TExtent>::value,
                              "The views and the extent are required to have the same dimensionality!");

                static_assert(std::is_same_v<Elem<TViewDst>, std::remove_const_t<SrcType>>,
                              "The source and the destination view are required to have the same element type!");

                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                auto bytes = std::size_t{};

                if constexpr(Dim<TExtent>::value == 1)
                    bytes = static_cast<std::size_t>(extent::getWidth(ext)) * SrcBytes;
                else if constexpr(Dim<TExtent>::value == 2)
                    bytes = static_cast<std::size_t>(extent::getWidth(ext) * extent::getHeight(ext)) * SrcBytes;
                else
                    bytes = static_cast<std::size_t>(extent::getWidth(ext) * extent::getHeight(ext) * extent::getDepth(ext)) * SrcBytes;

                return alpaka::detail::TaskCopySycl<SrcType>{std::make_shared<alpaka::detail::TaskCopySyclImpl<SrcType>>(getPtrNative(viewSrc), getPtrNative(viewDst), bytes)};
            }
        };

        //#############################################################################
        //! The SYCL device-to-host memory copy trait specialization.
        template<typename TDim, typename TPltf>
        struct CreateTaskMemcpy<TDim, DevCpu, DevGenericSycl<TPltf>>
        {
            //-----------------------------------------------------------------------------
            template<typename TExtent, typename TViewSrc, typename TViewDst>
            ALPAKA_FN_HOST static auto createTaskMemcpy(TViewDst & viewDst, TViewSrc const& viewSrc, TExtent const & ext)
            {
                using SrcType = Elem<std::remove_const_t<TViewSrc>>;
                constexpr auto SrcBytes = sizeof(SrcType);

                static_assert(!std::is_const<TViewDst>::value, "The destination view cannot be const!");

                static_assert(Dim<TViewDst>::value == Dim<std::remove_const_t<TViewSrc>>::value,
                              "The source and the destination view are required to have the same dimensionality!");

                static_assert(Dim<TViewDst>::value == Dim<TExtent>::value,
                              "The views and the extent are required to have the same dimensionality!");

                static_assert(std::is_same_v<Elem<TViewDst>, SrcType>,
                              "The source and the destination view are required to have the same element type!");

                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                auto bytes = std::size_t{};

                if constexpr(Dim<TExtent>::value == 1)
                    bytes = static_cast<std::size_t>(extent::getWidth(ext)) * SrcBytes;
                else if constexpr(Dim<TExtent>::value == 2)
                    bytes = static_cast<std::size_t>(extent::getWidth(ext) * extent::getHeight(ext)) * SrcBytes;
                else
                    bytes = static_cast<std::size_t>(extent::getWidth(ext) * extent::getHeight(ext) * extent::getDepth(ext)) * SrcBytes;

                return alpaka::detail::TaskCopySycl<SrcType>{std::make_shared<alpaka::detail::TaskCopySyclImpl<SrcType>>(getPtrNative(viewSrc), getPtrNative(viewDst), bytes)};
            }
        };

        //#############################################################################
        //! The SYCL device-to-device memory copy trait specialization.
        template<typename TDim, typename TPltfDst, typename TPltfSrc>
        struct CreateTaskMemcpy<TDim, DevGenericSycl<TPltfDst>, DevGenericSycl<TPltfSrc>>
        {
            //-----------------------------------------------------------------------------
            template<typename TExtent, typename TViewSrc, typename TViewDst>
            ALPAKA_FN_HOST static auto createTaskMemcpy(TViewDst & viewDst, TViewSrc const& viewSrc, TExtent const & ext)
            {
                using SrcType = Elem<std::remove_const_t<TViewSrc>>;
                constexpr auto SrcBytes = sizeof(SrcType);

                static_assert(!std::is_const<TViewDst>::value, "The destination view cannot be const!");

                static_assert(Dim<TViewDst>::value == Dim<std::remove_const_t<TViewSrc>>::value,
                              "The source and the destination view are required to have the same dimensionality!");

                static_assert(Dim<TViewDst>::value == Dim<TExtent>::value,
                              "The views and the extent are required to have the same dimensionality!");

                static_assert(std::is_same_v<Elem<TViewDst>, SrcType>,
                              "The source and the destination view are required to have the same element type!");

                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                auto bytes = std::size_t{};

                if constexpr(Dim<TExtent>::value == 1)
                    bytes = static_cast<std::size_t>(extent::getWidth(ext)) * SrcBytes;
                else if constexpr(Dim<TExtent>::value == 2)
                    bytes = static_cast<std::size_t>(extent::getWidth(ext) * extent::getHeight(ext)) * SrcBytes;
                else
                    bytes = static_cast<std::size_t>(extent::getWidth(ext) * extent::getHeight(ext) * extent::getDepth(ext)) * SrcBytes;

                return alpaka::detail::TaskCopySycl<SrcType>{std::make_shared<alpaka::detail::TaskCopySyclImpl<SrcType>>(getPtrNative(viewSrc), getPtrNative(viewDst), bytes)};
            }
        };
    }
}

#endif
