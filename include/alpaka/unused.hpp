/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

namespace alpaka
{
    /** Utility to mark variables as unused to avoid compiler warnings
     *
     * Using '[[maybe_unused]]' in function interfaces for arguments make the interface long and sometimes it is not
     * important that only the argument type is used within the function and not the instance itself.
     * This can be used to keep the function interfaces clean and readable.
     */
    inline constexpr void unused([[maybe_unused]] auto&&... values)
    {
    }
} // namespace alpaka
