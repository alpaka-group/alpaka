/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OACC_ENABLED

#if _OPENACC < 201306
    #error If ALPAKA_ACC_ANY_BT_OACC_ENABLED is set, the compiler has to support OpenACC xx or higher!
#endif

#include <alpaka/pltf/Traits.hpp>
#include <alpaka/dev/DevOacc.hpp>
#include <alpaka/core/Concepts.hpp>

#include <sstream>
#include <vector>

namespace alpaka
{
    namespace pltf
    {
        //#############################################################################
        //! The OpenAcc device platform.
        class PltfOacc :
            public concepts::Implements<ConceptPltf, PltfOacc>
        {
        public:
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST PltfOacc() = delete;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The OpenAcc device device type trait specialization.
            template<>
            struct DevType<
                pltf::PltfOacc>
            {
                using type = dev::DevOacc;
            };
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The OpenACC platform device count get trait specialization.
            template<>
            struct GetDevCount<
                pltf::PltfOacc>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDevCount()
                -> std::size_t
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    return static_cast<std::size_t>(::acc_get_num_devices(::acc_get_device_type()));
                }
            };

            //#############################################################################
            //! The OpenACC platform device get trait specialization.
            template<>
            struct GetDevByIdx<
                pltf::PltfOacc>
            {
                //-----------------------------------------------------------------------------
                //! \param devIdx device id, less than GetDevCount, will be set to omp_get_initial_device() if < 0
                ALPAKA_FN_HOST static auto getDevByIdx(
                    std::size_t devIdx)
                -> dev::DevOacc
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    std::size_t const devCount(pltf::getDevCount<pltf::PltfOacc>());
                    if(devIdx >= devCount)
                    {
                        std::stringstream ssErr;
                        ssErr << "Unable to return device handle for OpenACC device with index "
                            << devIdx << " because there are only " << devCount << " devices!";
                        throw std::runtime_error(ssErr.str());
                    }

                    return {static_cast<int>(devIdx)};
                }
            };
        }
    }
}

#endif
