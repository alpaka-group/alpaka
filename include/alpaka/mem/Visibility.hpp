/* Copyright 2023 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/Traits.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/mem/view/ViewAccessOps.hpp"
#include "alpaka/meta/ForEachType.hpp"
#include "alpaka/meta/TypeListOps.hpp"
#include "alpaka/platform/Traits.hpp"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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
    CREATE_MEM_VISIBILITY(MemVisibleCpuSycl);
    CREATE_MEM_VISIBILITY(MemVisibleGpuCudaRt);
    CREATE_MEM_VISIBILITY(MemVisibleGpuHipRt);
    CREATE_MEM_VISIBILITY(MemVisibleGpuSyclIntel);
#undef CREATE_MEM_VISIBILITY

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
                vs.push_back(TTYPE::get_name());
            }
        };
    } // namespace detail

    template<
        typename T,
        typename = std::enable_if_t<
            alpaka::isPlatform<std::decay_t<T>> || alpaka::isDevice<std::decay_t<T>>
            || alpaka::isAccelerator<std::decay_t<T>> || alpaka::internal::isView<std::decay_t<T>>>>
    inline std::string getMemVisiblityName()
    {
        using MemVisibilityType = typename alpaka::trait::MemVisibility<std::decay_t<T>>::type;
        if constexpr(alpaka::meta::isList<MemVisibilityType>)
        {
            std::vector<std::string> vs;
            alpaka::meta::forEachType<MemVisibilityType>(detail::AppendMemTypeName{}, vs);

            std::stringstream ss;
            ss << "<";
            for(std::size_t i = 0; i < vs.size(); ++i)
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
            return MemVisibilityType::get_name();
        }
    }

    template<typename TType>
    [[maybe_unused]] static std::string getMemVisiblityName(TType)
    {
        return getMemVisiblityName<TType>();
    }

    template<
        typename T,
        typename TBuf,
        typename = std::enable_if_t<
            (alpaka::isPlatform<std::decay_t<T>> || alpaka::isDevice<std::decay_t<T>>
             || alpaka::isAccelerator<std::decay_t<T>>) &&alpaka::internal::isView<std::decay_t<TBuf>>>>
    inline constexpr bool hasSameMemView()
    {
        if constexpr(alpaka::isDevice<std::decay_t<T>> || alpaka::isAccelerator<std::decay_t<T>>)
        {
            using Platform = alpaka::Platform<T>;
            return alpaka::meta::Contains<
                typename alpaka::trait::MemVisibility<TBuf>::type,
                typename alpaka::trait::MemVisibility<Platform>::type>::value;
        }
        else
        {
            return alpaka::meta::Contains<
                typename alpaka::trait::MemVisibility<TBuf>::type,
                typename alpaka::trait::MemVisibility<T>::type>::value;
        }
        ALPAKA_UNREACHABLE({});
    }

    template<typename TDev, typename TBuf>
    inline constexpr bool hasSameMemView(TDev&, TBuf&)
    {
        return hasSameMemView<std::decay_t<TDev>, std::decay_t<TBuf>>();
    }

    namespace detail
    {
        template<typename T, typename = void>
        struct MemVisibilityHelper
        {
            using type = typename alpaka::trait::MemVisibility<T>::type;
        };

        template<typename T>
        struct MemVisibilityHelper<
            T,
            std::enable_if_t<alpaka::isDevice<std::decay_t<T>> || alpaka::isAccelerator<std::decay_t<T>>>>
        {
            using type = typename alpaka::trait::MemVisibility<alpaka::Platform<std::decay_t<T>>>::type;
        };
    } // namespace detail

    template<typename T>
    using MemVisibility = typename alpaka::detail::MemVisibilityHelper<std::decay_t<T>>::type;

    template<typename T>
    using MemVisibilityTypeList = alpaka::meta::toTuple<alpaka::MemVisibility<std::decay_t<T>>>;
} // namespace alpaka
