/* Copyright 2023 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/meta/ForEachType.hpp"
#include "alpaka/meta/IsTuple.hpp"
#include "alpaka/meta/TypeListOps.hpp"

#include <iostream>
#include <sstream>
#include <string>

#define CREATE_MEM_VISIBILITY(mem_name)                                                                               \
    struct mem_name                                                                                                   \
    {                                                                                                                 \
        static std::string name()                                                                                     \
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
        //! \tparam TType which implements the trait
        template<typename TType>
        struct MemVisibility;
    } // namespace trait

    namespace detail
    {
        struct AppendMemTypeName
        {
            template<typename TTYPE>
            void operator()(std::vector<std::string>& vs)
            {
                vs.push_back(TTYPE::name());
            }
        };
    } // namespace detail

    template<typename TType>
    static std::string getMemVisiblityName()
    {
        using MemVisibilityType = typename alpaka::trait::MemVisibility<std::decay_t<TType>>::type;
        if constexpr(alpaka::meta::isTuple<MemVisibilityType>())
        {
            std::vector<std::string> vs;
            alpaka::meta::forEachType<MemVisibilityType>(detail::AppendMemTypeName{}, vs);

            std::stringstream ss;
            ss << "<";
            for(auto i = 0; i < vs.size(); ++i)
            {
                if(i == (vs.size() - 1))
                {
                    ss << vs[i] << ">";
                }
                else
                {
                    ss << vs[i] << ", ";
                }
            }
            return ss.str();
        }
        else
        {
            return MemVisibilityType::name();
        }
    }

    template<typename TType>
    static std::string getMemVisiblityName(TType)
    {
        return getMemVisiblityName<TType>();
    }

    template<typename TDev, typename TBuf>
    inline constexpr bool hasSameMemView()
    {
        return alpaka::meta::Contains<
            typename alpaka::trait::MemVisibility<TBuf>::type,
            typename alpaka::trait::MemVisibility<TDev>::type>::value;
    }

    template<typename TDev, typename TBuf>
    inline constexpr bool hasSameMemView(TDev, TBuf)
    {
        return hasSameMemView<std::decay_t<TDev>, std::decay_t<TBuf>>();
    }
} // namespace alpaka
