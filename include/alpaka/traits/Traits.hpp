/* Copyright 2022 Antonio Di Pilato
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>

namespace alpaka
{
    //! The common traits.
    namespace traits
    {
        //! The native handle trait.
        template<typename TImpl, typename TSfinae = void>
        struct NativeHandle
        {
            static auto getNativeHandle(TImpl const&)
            {
                static_assert(!sizeof(TImpl), "This type does not have a native handle!");
                return 0;
            }
        };
    } // namespace traits

    //! Get the native handle of the alpaka object.
    //! It will return the alpaka object handle if there is any, otherwise it generates a compile time error.
    template<typename TImpl>
    ALPAKA_FN_HOST auto getNativeHandle(TImpl const& impl)
    {
        return traits::NativeHandle<TImpl>::getNativeHandle(impl);
    }
} // namespace alpaka
