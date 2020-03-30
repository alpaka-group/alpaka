/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_CPU_BT_OMP4_ENABLED

#if _OPENMP < 201307
    #error If ALPAKA_ACC_CPU_BT_OMP4_ENABLED is set, the compiler has to support OpenMP 4.0 or higher!
#endif

#include <alpaka/pltf/Traits.hpp>
#include <alpaka/dev/DevOmp4.hpp>
#include <alpaka/core/Concepts.hpp>

#include <sstream>
#include <vector>
#include <limits>

namespace alpaka
{
    namespace pltf
    {
        //#############################################################################
        //! The CPU device platform.
        class PltfOmp4 :
            public concepts::Implements<ConceptPltf, PltfOmp4>
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

                    return static_cast<std::size_t>(::omp_get_num_devices());
                }
            };

            //#############################################################################
            //! The CPU platform device get trait specialization.
            template<>
            struct GetDevByIdx<
                pltf::PltfOmp4>
            {
                //-----------------------------------------------------------------------------
                //! \param devIdx device id, less than GetDevCount, will be set to omp_get_initial_device() if < 0
                ALPAKA_FN_HOST static auto getDevByIdx(
                    std::size_t devIdx)
                -> dev::DevOmp4
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    std::size_t const devCount(static_cast<std::size_t>(pltf::getDevCount<pltf::PltfOmp4>()));
                    int devIdxOmp4 = static_cast<int>(devIdx);
                    if(devIdx >= devCount)
                    { // devIdx param must be unsigned, take take this case to use the initial device
                        devIdxOmp4 = ::omp_get_initial_device();
                    }

                    return {devIdxOmp4};
                }
            };
        }
    }
}

#endif
