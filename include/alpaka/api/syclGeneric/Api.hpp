/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/concepts.hpp"

#include <memory>
#include <string>

namespace alpaka
{
    namespace api
    {
        template<typename T_ApiInterface>
        // NOLINT because API tags such as OneApi, Cuda, Hip, and Host are aggregates.
        // Making this CRTP constructor private would prevent their aggregate
        // initialization, requiring explicit constructors and an unnecessary API change.
        // NOLINTNEXTLINE(bugprone-crtp-constructor-accessibility)
        struct GenericSycl : detail::ApiBase
        {
            using element_type = T_ApiInterface;

            auto get() const
            {
                return static_cast<T_ApiInterface const*>(this);
            }

            void _()
            {
                static_assert(concepts::Api<GenericSycl<T_ApiInterface>>);
            }

            static std::string getName()
            {
                return "GenericSycl";
            }
        };
    } // namespace api
} // namespace alpaka
