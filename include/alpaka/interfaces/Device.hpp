/**
* Copyright 2014 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of either the GNU General Public License or
* the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License and the GNU Lesser General Public License
* for more details.
*
* You should have received a copy of the GNU General Public License
* and the GNU Lesser General Public License along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/core/Vec.hpp>  // alpaka::vec

#include <vector>               // std::vector
#include <string>               // std::string
#include <cstddef>              // std::size_t

namespace alpaka
{
    namespace device
    {
        //#############################################################################
        //! The interface of a device handle.
        //#############################################################################
        struct DeviceProperties
        {
            std::string m_sName;
            std::size_t m_uiBlockKernelSizeMax;
            vec<3u> m_v3uiBlockKernelSizePerDimMax;
            vec<3u> m_v3uiGridBlockSizePerDimMax;
            std::size_t m_uiStreamCount;
            std::size_t m_uiExecutionUnitCount;
            std::size_t m_uiGlobalMemorySizeBytes;
            //std::size_t m_uiClockFrequencyHz;
        };

        namespace detail
        {
            //#############################################################################
            //! The template of a device handle.
            //#############################################################################
            template<typename TDeviceHandle>
            class IDeviceHandle :
                private TDeviceHandle
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST IDeviceHandle() = default;
                //-----------------------------------------------------------------------------
                //! Copy-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST IDeviceHandle(IDeviceHandle const &) = default;
                //-----------------------------------------------------------------------------
                //! Move-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST IDeviceHandle(IDeviceHandle &&) = default;
                //-----------------------------------------------------------------------------
                //! Assignment-operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST IDeviceHandle & operator=(IDeviceHandle const &) = default;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST ~IDeviceHandle() noexcept = default;

                //-----------------------------------------------------------------------------
                //! \return The device properties.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DeviceProperties getProperties() const
                {
                    this->TDeviceHandle::getProperties();
                }
            };
        }

        //#############################################################################
        //! The template of a device handle.
        //#############################################################################
        template<typename TAcc>
        class DeviceHandle;

        namespace detail
        {
            //#############################################################################
            //! The interface for device selection.
            //#############################################################################
            template<typename TDeviceManager>
            class IDeviceManager :
                private TDeviceManager
            {
            public:
                using TDeviceHandle = typename decltype(TDeviceManager::getCurrentDeviceHandle());

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
                ALPAKA_FCT_HOST static TDeviceHandle getDeviceHandleByIndex()
                {
                    return TDeviceManager::getDeviceHandleByIndex()
                }
                //-----------------------------------------------------------------------------
                //! \return The number handles to all devices available.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static std::vector<TDeviceHandle> getDeviceHandles()
                {
                    std::vector<TDeviceHandle> vDeviceHandles;

                    std::size_t const uiDeviceCount(getDeviceCount());
                    for(std::size_t uiDeviceIndex(0); uiDeviceIndex < uiDeviceCount; ++uiDeviceIndex)
                    {
                        vDeviceHandles.push_back(getDeviceHandle(uiDeviceIndex));
                    }

                    return vDeviceHandles;
                }
                //-----------------------------------------------------------------------------
                //! \return The handle to the currently used device.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static TDeviceHandle getCurrentDeviceHandle()
                {
                    return TDeviceManager::getCurrentDeviceHandle();
                }
                //-----------------------------------------------------------------------------
                //! Sets the device to use with this accelerator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static void setCurrentDevice(TDeviceHandle const & deviceHandle)
                {
                    TDeviceManager::setCurrentDevice(deviceHandle);
                }
            };
        }

        //#############################################################################
        //! The template of a device manager.
        //#############################################################################
        template<typename TAcc>
        class DeviceManager;
    }
}
