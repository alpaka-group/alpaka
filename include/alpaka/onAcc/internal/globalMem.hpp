/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/api.hpp"
#include "alpaka/concepts/types.hpp"
#include "alpaka/core/PP.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/mem/MdSpan.hpp"
#include "alpaka/mem/MdSpanArray.hpp"

/** @file
 * @brief global device memory implementation for all APIs
 *
 * We need many preprocessor macros to handle the device global feature.
 * The reason is that we would like to have the possibility to create a variable where the same name can be used on all
 * devices. Each device will have its own instance of memory and via a global instance GlobalDeviceMemoryWrapper we
 * redirect queries to the corresponding instance of memory based on the alpaka API.
 *
 * OneAPI Sycl is the only API which does not allow querying the device pointer of a global variable from the host.
 * That's why we have special implementations of onHost::memcpy() which using the device global object directly.
 * That's also the reason why we can not use the global memory for onHost::fill() or onHost::memset().
 */
namespace alpaka::onHost::internal
{
    struct MemcpyDeviceGlobal;
} // namespace alpaka::onHost::internal

/** Create a device global variable for the API host */
#define ALPAKA_DEVICE_GLOBAL_DATA_HOST(attributes, dataType, name, ...)                                               \
    namespace alpaka_onHost                                                                                           \
    {                                                                                                                 \
        [[maybe_unused]] attributes alpaka::onAcc::internal::GlobalDeviceMemoryDataWrapper<                           \
            ALPAKA_PP_REMOVE_BRACKETS(dataType)> name __VA_OPT__({__VA_ARGS__});                                      \
    }

/** Create a forward declaration of a device global variable for the API host */
#define ALPAKA_DEVICE_GLOBAL_DATA_HOST_EXTERN(attributes, dataType, name)                                             \
    namespace alpaka_onHost                                                                                           \
    {                                                                                                                 \
        extern attributes alpaka::onAcc::internal::GlobalDeviceMemoryDataWrapper<ALPAKA_PP_REMOVE_BRACKETS(dataType)> \
            name;                                                                                                     \
    }

#if ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP
/** Create a device global variable for the API cuda/hip */
#    define ALPAKA_DEVICE_GLOBAL_DATA_CUDA_HIP(attributes, dataType, name, ...)                                       \
        namespace alpaka_onAccCudaHip                                                                                 \
        {                                                                                                             \
            __device__ attributes                                                                                     \
                alpaka::onAcc::internal::GlobalDeviceMemoryDataWrapper<ALPAKA_PP_REMOVE_BRACKETS(dataType)>           \
                    name __VA_OPT__({__VA_ARGS__});                                                                   \
        }

/** Access operator for usage in AlpakaGlobalStorage for API cuda/hip */
#    define ALPAKA_DEVICE_GLOBAL_GET_CUDA_HIP(attributes, dataType, name, ...)                                        \
        template<typename T_Api>                                                                                      \
        requires(std::is_same_v<alpaka::api::Cuda, T_Api> || std::is_same_v<alpaka::api::Hip, T_Api>)                 \
        constexpr auto& get(T_Api) const                                                                              \
        {                                                                                                             \
            return alpaka_onAccCudaHip::name.value;                                                                   \
        }                                                                                                             \
        template<typename T_Api>                                                                                      \
        requires(std::is_same_v<alpaka::api::Cuda, T_Api> || std::is_same_v<alpaka::api::Hip, T_Api>)                 \
        constexpr auto& getHandle(T_Api) const                                                                        \
        {                                                                                                             \
            return alpaka_onAccCudaHip::name.value;                                                                   \
        }

#else
#    define ALPAKA_DEVICE_GLOBAL_DATA_CUDA_HIP(attributes, dataType, name, ...)
#    define ALPAKA_DEVICE_GLOBAL_GET_CUDA_HIP(attributes, dataType, name, ...)
#endif

#if ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP
/* Define the device external symbol only if relocatable device code is enabled. nvcc is changing the keyword 'extern'
 * to static in case rdc is disabled which results into redefinition compile errors. To make HIP and CUDA behave equal
 * we do not expose the symbol for HIP as well in case rdc is disabled.
 */
#    if defined(__CUDACC_RDC__) || defined(__CLANG_RDC__)
#        define ALPAKA_DEVICE_GLOBAL_DATA_CUDA_HIP_EXTERN(attributes, dataType, name)                                 \
            namespace alpaka_onAccCudaHip                                                                             \
            {                                                                                                         \
                extern __device__ attributes                                                                          \
                    alpaka::onAcc::internal::GlobalDeviceMemoryDataWrapper<ALPAKA_PP_REMOVE_BRACKETS(dataType)>       \
                        name;                                                                                         \
            }
#    else
/** Create a forward declaration of a device global variable for the API cuda/hip */
#        define ALPAKA_DEVICE_GLOBAL_DATA_CUDA_HIP_EXTERN(attributes, dataType, name)
#    endif
#else
#    define ALPAKA_DEVICE_GLOBAL_DATA_CUDA_HIP_EXTERN(attributes, dataType, name)
#endif

#if ALPAKA_LANG_ONEAPI
/** Create a device global variable for the API oneApi */
#    define ALPAKA_DEVICE_GLOBAL_DATA_ONEAPI(attributes, dataType, name, ...)                                         \
        namespace alpaka_onAccOneAPI                                                                                  \
        {                                                                                                             \
            [[maybe_unused]] attributes sycl::ext::oneapi::experimental::device_global<                               \
                alpaka::onAcc::internal::GlobalDeviceMemoryDataWrapper<ALPAKA_PP_REMOVE_BRACKETS(dataType)>> name     \
                __VA_OPT__(                                                                                           \
                    {alpaka::onAcc::internal::GlobalDeviceMemoryDataWrapper<ALPAKA_PP_REMOVE_BRACKETS(dataType)>{     \
                        __VA_ARGS__}});                                                                               \
        }

/** Create a forward declaration of a device global variable for the API oneApi */
#    define ALPAKA_DEVICE_GLOBAL_DATA_ONEAPI_EXTERN(attributes, dataType, name)                                       \
        namespace alpaka_onAccOneAPI                                                                                  \
        {                                                                                                             \
            extern attributes sycl::ext::oneapi::experimental::device_global<                                         \
                alpaka::onAcc::internal::GlobalDeviceMemoryDataWrapper<ALPAKA_PP_REMOVE_BRACKETS(dataType)>>          \
                name;                                                                                                 \
        }

/** Access operator for usage in AlpakaGlobalStorage for API oneApi */
#    define ALPAKA_DEVICE_GLOBAL_GET_ONEAPI(attributes, dataType, name, ...)                                          \
        template<typename T_Api>                                                                                      \
        requires(std::is_same_v<alpaka::api::OneApi, T_Api>)                                                          \
        constexpr auto& get(T_Api) const                                                                              \
        {                                                                                                             \
            return alpaka_onAccOneAPI::name.get().value;                                                              \
        }                                                                                                             \
        template<typename T_Api>                                                                                      \
        requires(std::is_same_v<alpaka::api::OneApi, T_Api>)                                                          \
        constexpr auto& getHandle(T_Api) const                                                                        \
        {                                                                                                             \
            return alpaka_onAccOneAPI::name;                                                                          \
        }
#else
#    define ALPAKA_DEVICE_GLOBAL_DATA_ONEAPI(attributes, dataType, name, ...)
#    define ALPAKA_DEVICE_GLOBAL_DATA_ONEAPI_EXTERN(attributes, dataType, name)
#    define ALPAKA_DEVICE_GLOBAL_GET_ONEAPI(attributes, dataType, name, ...)
#endif

namespace alpaka::onAcc::internal
{
    /** Helper class to wrap device global memory data.
     *
     * The reason why this wrapper is required is that SYCL oneAPI is using a special type which does not support C
     * array initialization. All arguments passed to the wrapper constructor are forwarded to the data member.
     */
    template<typename T>
    struct GlobalDeviceMemoryDataWrapper
    {
        constexpr GlobalDeviceMemoryDataWrapper(auto const&... args) : value{ALPAKA_FORWARD(args)...}
        {
        }

        T value;

        T* data()
        {
            return &value;
        }

        T const* data() const
        {
            return &value;
        }
    };

    /** Specialization of GlobalDeviceMemoryDataWrapper for C static arrays.
     *
     * This specialization is required because C static arrays cannot have a constructor.
     * Therefore, the data member is initialized directly.
     */
    template<alpaka::concepts::CStaticArray T>
    struct GlobalDeviceMemoryDataWrapper<T>
    {
        T value;
        using value_type = std::remove_all_extents_t<T>;

        value_type* data()
        {
            return reinterpret_cast<value_type*>(&value);
        }

        value_type const* data() const
        {
            return reinterpret_cast<value_type const*>(&value);
        }
    };

    /** Helper class to provide access to device global memory variables */
    template<typename T_Storage, typename T_Type>
    struct GlobalDeviceMemoryWrapper : private T_Storage
    {
    private:
        friend struct onHost::internal::MemcpyDeviceGlobal;

        /** Get the handle to call native API specific memcopy for global device memory operation
         *
         * @attention This method is for internal usage only.
         *
         * @return type depends on the native API e.g Cuda, OneApi, ...
         */
        template<alpaka::concepts::Api T_Api>
        constexpr decltype(auto) getHandle(T_Api api) const
        {
            return T_Storage::getHandle(api);
        }

    public:
        using type = T_Type;

        constexpr decltype(auto) get() const
        {
            return T_Storage::get(thisApi());
        }

        constexpr decltype(auto) get() const requires(std::is_array_v<type>)
        {
            // a static array of type C also uses size_t to define its length
            return alpaka::MdSpanArray<type, size_t>{T_Storage::get(thisApi())};
        }

        constexpr operator type&()
        {
            return T_Storage::get(thisApi());
        }

        constexpr operator type const&() const
        {
            return T_Storage::get(thisApi());
        }
    };
} // namespace alpaka::onAcc::internal
