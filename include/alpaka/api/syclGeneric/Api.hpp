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
        // a private constructor of the base class would require OneApi to define its own constructor, but all API
        // tags, e.g. Cuda, Hip and Host, are aggregates without one
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
