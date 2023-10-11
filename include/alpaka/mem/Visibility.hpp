/* Copyright 2023 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/meta/TypeListOps.hpp"

#include <iostream>

#define CREATE_MEM_VISIBILITY(mem_name)                                                                               \
    struct mem_name                                                                                                   \
    {                                                                                                                 \
        static std::string get_name()                                                                                 \
        {                                                                                                             \
            return #mem_name;                                                                                         \
        }                                                                                                             \
    }

namespace alpaka
{
    CREATE_MEM_VISIBILITY(MemVisibleCPU);
    CREATE_MEM_VISIBILITY(MemVisibleFpgaSyclIntel);
    CREATE_MEM_VISIBILITY(MemVisibleGenericSycl);
    CREATE_MEM_VISIBILITY(MemVisibleGpuCudaRt);
    CREATE_MEM_VISIBILITY(MemVisibleGpuHipRt);
    CREATE_MEM_VISIBILITY(MemVisibleGpuSyclIntel);

    namespace trait
    {
        //! Get memory visibility from a type.
        //! Normally it is acc or buffer type.
        //!
        //! \tparam Type which implements the trait
        //! \return Memory visibility type
        template<typename TType>
        struct MemVisibility;
    } // namespace trait

    template<typename TAcc, typename TBuf>
    inline constexpr bool hasSameMemView()
    {
        return alpaka::meta::Contains<
            typename alpaka::trait::MemVisibility<TBuf>::type,
            typename alpaka::trait::MemVisibility<TAcc>::type>::value;
    }
} // namespace alpaka
