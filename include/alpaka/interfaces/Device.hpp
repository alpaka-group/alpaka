/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/core/Vec.hpp>  // alpaka::Vec

#include <vector>               // std::vector
#include <string>               // std::string
#include <cstddef>              // std::size_t

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The device management functionality.
    //-----------------------------------------------------------------------------
    namespace device
    {
        //#############################################################################
        //! The device properties.
        //#############################################################################
        struct DeviceProperties
        {
            std::string m_sName;                    //!< The name.
            std::size_t m_uiMultiProcessorCount;    //!< The number of multiprocessors.
            std::size_t m_uiBlockKernelsCountMax;   //!< The maximum number of kernels in a block.
            Vec<3u> m_v3uiBlockKernelsExtentMax;    //!< The maximum number of kernels in each dimension of a block.
            Vec<3u> m_v3uiGridBlocksExtentMax;      //!< The maximum number of blocks in each dimension of the grid.
            std::size_t m_uiGlobalMemorySizeBytes;  //!< Size of the global device memory in bytes.
            //std::size_t m_uiSharedMemorySizeBytes;  //!< Size of the available block shared memory in bytes. 
            //std::size_t m_uiMaxClockFrequencyHz;    //!< Maximum clock frequency of the device in Hz.
        };

        namespace detail
        {
            //#############################################################################
            //! The device handle interface.
            //#############################################################################
            template<typename TDevice>
            class IDevice :
                protected TDevice
            {
            protected:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST IDevice() = default;
            public:
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST IDevice(IDevice const &) = default;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST IDevice(IDevice &&) = default;
                //-----------------------------------------------------------------------------
                //! Assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST IDevice & operator=(IDevice const &) = default;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST virtual ~IDevice() noexcept = default;

                //-----------------------------------------------------------------------------
                //! \return The device properties.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DeviceProperties getProperties() const
                {
                    return this->TDevice::getProperties();
                }
            };
        }

        //#############################################################################
        //! The abstract device handle.
        //#############################################################################
        template<typename TAcc>
        class Device;

        namespace detail
        {
            //#############################################################################
            //! The abstract thread device waiter.
            //#############################################################################
            template<typename TStream, typename TSfinae = void>
            struct ThreadWaitDevice;
        }

        //#############################################################################
        //! Waits for the device to complete all of its work.
        //#############################################################################
        template<typename TAcc>
        ALPAKA_FCT_HOST void wait(Device<TAcc> const & device)
        {
            detail::ThreadWaitDevice<Device<TAcc>>{device};
        }

        namespace detail
        {
            //#############################################################################
            //! The device manager interface.
            //#############################################################################
            template<typename TDeviceManager>
            class IDeviceManager :
                private TDeviceManager
            {
            public:
                using Device = decltype(TDeviceManager::getCurrentDevice());

            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST IDeviceManager() = delete;

                //-----------------------------------------------------------------------------
                //! \return The number of devices available.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static std::size_t getDeviceCount()
                {
                    return TDeviceManager::getDeviceCount();
                }
                //-----------------------------------------------------------------------------
                //! \return The handle to the device with the given index.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static Device getDeviceByIndex()
                {
                    return TDeviceManager::getDeviceByIndex();
                }
                //-----------------------------------------------------------------------------
                //! \return The number handles to all devices available.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static std::vector<Device> getDevices()
                {
                    std::vector<TDevice> vDevices;

                    std::size_t const uiDeviceCount(getDeviceCount());
                    for(std::size_t uiDeviceIndex(0); uiDeviceIndex < uiDeviceCount; ++uiDeviceIndex)
                    {
                        vDevices.push_back(getDeviceByIndex(uiDeviceIndex));
                    }

                    return vDevices;
                }
                //-----------------------------------------------------------------------------
                //! \return The handle to the currently used device.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static Device getCurrentDevice()
                {
                    return TDeviceManager::getCurrentDevice();
                }
                //-----------------------------------------------------------------------------
                //! Sets the device to use with this accelerator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static void setCurrentDevice(Device const & device)
                {
                    TDeviceManager::setCurrentDevice(device);
                }
            };
        }

        //#############################################################################
        //! The abstract device manager.
        //#############################################################################
        template<typename TAcc>
        class DeviceManager;
    }
}
