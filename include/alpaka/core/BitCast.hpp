/* Copyright 2025 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cstring>
#include <type_traits>

namespace alpaka::core
{
    //! From https://en.cppreference.com/w/cpp/numeric/bit_cast.html
    template<class To, class From>
    std::enable_if_t<
        sizeof(To) == sizeof(From) && std::is_trivially_copyable_v<From> && std::is_trivially_copyable_v<To>,
        To>
    bit_cast(From const& src) noexcept
    {
        std::aligned_storage_t<sizeof(To), alignof(To)> dst;
        std::memcpy(&dst, &src, sizeof(To));
        return *reinterpret_cast<To*>(&dst);
    }

} // namespace alpaka::core
