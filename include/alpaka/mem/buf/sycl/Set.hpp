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
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dev/DevGenericSycl.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/extent/Traits.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/core/Assert.hpp>
#include <alpaka/core/Sycl.hpp>

#include <CL/sycl.hpp>

#include <vector>

namespace alpaka
{
    namespace detail
    {
        template <typename TElem>
        struct TaskSetSyclImpl
        {
            TaskSetSyclImpl(TElem* p, int val, std::size_t size)
            : ptr{p}, value{val}, bytes{size}
            {}

            TaskSetSyclImpl(TaskSetSyclImpl const&) = delete;
            auto operator=(TaskSetSyclImpl const&) -> TaskSetSyclImpl& = delete;
            TaskSetSyclImpl(TaskSetSyclImpl&&) = default;
            auto operator=(TaskSetSyclImpl&&) -> TaskSetSyclImpl& = default;
            ~TaskSetSyclImpl() = default;

            TElem* ptr;
            int value;
            std::size_t bytes;
            std::vector<cl::sycl::event> dependencies = {};
            std::shared_mutex mutex{};
        };

        //#############################################################################
        //! The SYCL memory set trait.
        template<typename TElem>
        struct TaskSetSycl
        {
            //-----------------------------------------------------------------------------
            auto operator()(cl::sycl::handler& cgh) -> void
            {
                cgh.depends_on(pimpl->dependencies);
                cgh.memset(pimpl->ptr, pimpl->value, pimpl->bytes);
            }

            std::shared_ptr<TaskSetSyclImpl<TElem>> pimpl;
            // Distinguish from non-alpaka types (= host tasks)
            static constexpr auto is_sycl_enqueueable = true;
        };
    }

    namespace traits
    {
        //#############################################################################
        //! The SYCL device memory set trait specialization.
        template<typename TDim, typename TPltf>
        struct CreateTaskMemset<TDim, DevGenericSycl<TPltf>>
        {
            //-----------------------------------------------------------------------------
            template<typename TExtent, typename TView>
            ALPAKA_FN_HOST static auto createTaskMemset(TView & view, std::uint8_t const& byte, TExtent const& ext)
            {
                using Type = Elem<TView>;
                constexpr auto TypeBytes = sizeof(Type);

                auto bytes = std::size_t{};
                if constexpr(Dim<TExtent>::value == 1)
                    bytes = static_cast<std::size_t>(extent::getWidth(ext)) * TypeBytes;
                else if constexpr(Dim<TExtent>::value == 2)
                    bytes = static_cast<std::size_t>(extent::getWidth(ext) * extent::getHeight(ext)) * TypeBytes;
                else
                    bytes = static_cast<std::size_t>(extent::getWidth(ext) * extent::getHeight(ext) * extent::getDepth(ext)) * TypeBytes;

                return alpaka::detail::TaskSetSycl<Type>{std::make_shared<alpaka::detail::TaskSetSyclImpl<Type>>(getPtrNative(view), static_cast<int>(byte), bytes)};
            }
        };
    }
}

#endif
