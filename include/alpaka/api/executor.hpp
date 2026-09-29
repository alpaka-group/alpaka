/* Copyright 2024 René Widera, Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/cuda/executor.hpp"
#include "alpaka/api/hip/executor.hpp"
#include "alpaka/api/host/executor.hpp"
#include "alpaka/api/oneApi/executor.hpp"
#include "alpaka/api/trait.hpp"

#include <string>

namespace alpaka
{
    namespace exec
    {
        /** Automatic executor selection
         *
         * If this executor is used in alpaka interfaces, the best fitting available executor will automatically
         * select. The selection is often based on the device or queue provided in the interfaces.
         */
        struct AnyExecutor
        {
            static std::string getName()
            {
                return "AnyExecutor";
            }
        };

        /** @copydoc AnyExecutor */
        constexpr AnyExecutor anyExecutor;
    } // namespace exec

    namespace trait
    {
        template<>
        struct IsExecutor<exec::AnyExecutor> : std::true_type
        {
        };
    } // namespace trait
} // namespace alpaka
