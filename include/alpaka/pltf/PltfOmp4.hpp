/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/pltf/Traits.hpp>
#include <alpaka/dev/DevOmp4.hpp>

#include <sstream>
#include <vector>

namespace alpaka
{
    namespace pltf
    {
        //#############################################################################
        //! The CPU device platform.
        class PltfOmp4
        {
        public:
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST PltfOmp4() = delete;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU device device type trait specialization.
            template<>
            struct DevType<
                pltf::PltfOmp4>
            {
                using type = dev::DevOmp4;
            };
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU platform device count get trait specialization.
            template<>
            struct GetDevCount<
                pltf::PltfOmp4>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDevCount()
                -> std::size_t
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    return 1;
                }
            };

            //#############################################################################
            //! The CPU platform device get trait specialization.
            template<>
            struct GetDevByIdx<
                pltf::PltfOmp4>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDevByIdx(
                    std::size_t const & devIdx)
                -> dev::DevOmp4
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    std::size_t const devCount(pltf::getDevCount<pltf::PltfOmp4>());
                    if(devIdx >= devCount)
                    {
                        std::stringstream ssErr;
                        ssErr << "Unable to return device handle for CPU device with index " << devIdx << " because there are only " << devCount << " devices!";
                        throw std::runtime_error(ssErr.str());
                    }

                    return {};
                }
            };
        }
    }
}
