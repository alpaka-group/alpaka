/* Copyright 2024 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <tuple>
#include <type_traits>

// copied from https://stackoverflow.com/a/51073558/22035743
namespace alpaka::meta
{
    template<typename T>
    struct IsTuple : std::false_type
    {
    };

    template<typename... U>
    struct IsTuple<std::tuple<U...>> : std::true_type
    {
    };

    template<typename T>
    constexpr bool isTuple()
    {
        return IsTuple<std::decay_t<T>>::value;
    }

} // namespace alpaka::meta
