/* Copyright 2024 René Widera, Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/concepts.hpp"
#include "alpaka/onHost/Device.hpp"
#include "alpaka/onHost/DeviceSpec.hpp"
#include "alpaka/utility.hpp"

namespace alpaka::onHost
{
    template<alpaka::concepts::Api T_Api, alpaka::concepts::DeviceKind T_DeviceKind>
    struct DeviceSelector
    {
    public:
        static_assert(
            DeviceSpec<T_Api, T_DeviceKind>::isValid(),
            "Invalid combination of device kind and api. The api does not know how to talk to the device or the "
            "required dependencies to enable the api are not fulfilled.");

        constexpr DeviceSelector(alpaka::concepts::DeviceSpec auto deviceSpec)
            : m_platform(internal::makePlatform(getApi(deviceSpec), getDeviceKind(deviceSpec)))
            , m_deviceSpec(deviceSpec)
        {
        }

        constexpr DeviceSelector(T_Api api, T_DeviceKind deviceKind) : DeviceSelector(DeviceSpec{api, deviceKind})
        {
        }

        /** Get the number of available devices for the given api and device kind.
         *
         * @attention In case the compiler flags you used to build your application were wrong, kernels for the given
         * deviceKind cannot be built and the number of available devices will be zero. This can happen, e.g., if you
         * compile for OneAPI SYCL: you can compile the application, but whether you can run on a device is evaluated
         * at runtime.
         *
         * @return number of devices
         */
        uint32_t getDeviceCount() const
        {
            return internal::GetDeviceCount::Op<ALPAKA_TYPEOF(*m_platform.get())>{}(*m_platform.get());
        }

        bool isAvailable() const
        {
            return getDeviceCount() != 0;
        }

        DeviceProperties getDeviceProperties(uint32_t idx) const
        {
            return internal::GetDeviceProperties::Op<ALPAKA_TYPEOF(*m_platform.get())>{}(*m_platform.get(), idx);
        }

        /** Get a device
         *
         * An runtime exception is thrown if the device is not usable e.g. if a CUDA device is set to access mode
         * "single process only".
         *
         * @param idx device index (range [0;number of devices), invalid index will throw an exception
         * @return @see onHost::Device
         */
        auto makeDevice(uint32_t idx)
        {
            return Device{internal::MakeDevice::Op<ALPAKA_TYPEOF(*m_platform.get())>{}(*m_platform.get(), idx)};
        }

    private:
        ALPAKA_TYPEOF(internal::makePlatform(T_Api{}, T_DeviceKind{})) m_platform;
        DeviceSpec<T_Api, T_DeviceKind> m_deviceSpec;
    };

    template<alpaka::concepts::DeviceSpec T_DeviceSpec>
    DeviceSelector(T_DeviceSpec const& deviceSpec)
        -> DeviceSelector<ALPAKA_TYPEOF(getApi(deviceSpec)), ALPAKA_TYPEOF(getDeviceKind(deviceSpec))>;

    /** create DeviceSelector to get access to devices */
    template<alpaka::concepts::DeviceSpec T_DeviceSpec>
    inline auto makeDeviceSelector(T_DeviceSpec deviceSpec)
    {
        return DeviceSelector{deviceSpec};
    }

    /** create a DeviceSelector to get access to devices */
    inline auto makeDeviceSelector(alpaka::concepts::Api auto api, alpaka::concepts::DeviceKind auto deviceKind)
    {
        return DeviceSelector{api, deviceKind};
    }

    template<typename T_DeferEvaluation = void>
    inline auto makeHostDevice()
    {
        return DeviceSelector{
            std::conditional_t<std::is_same_v<T_DeferEvaluation, bool>, api::Host, api::Host>{},
            deviceKind::cpu}
            .makeDevice(0);
    }
} // namespace alpaka::onHost

namespace alpaka::api
{
    /** Shorthand operator to create a DeviceSelector.
     *
     * Using this operator is equal to call onHost::makeDeviceSelector(api, deviceKind);
     */
    inline auto operator+(alpaka::concepts::Api auto api, alpaka::concepts::DeviceKind auto deviceKind)
        -> onHost::DeviceSelector<ALPAKA_TYPEOF(api), ALPAKA_TYPEOF(deviceKind)>
    {
        return onHost::DeviceSelector{api, deviceKind};
    }

    /** Shorthand operator to create a DeviceSelector.
     *
     * Using this operator is equal to call onHost::makeDeviceSelector(api, deviceKind);
     */
    inline auto operator+(alpaka::concepts::DeviceKind auto deviceKind, alpaka::concepts::Api auto api)
        -> onHost::DeviceSelector<ALPAKA_TYPEOF(api), ALPAKA_TYPEOF(deviceKind)>
    {
        return onHost::DeviceSelector{api, deviceKind};
    }
} // namespace alpaka::api
