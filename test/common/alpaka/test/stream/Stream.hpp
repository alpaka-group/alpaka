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

#include <alpaka/alpaka.hpp>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The test specifics.
    //-----------------------------------------------------------------------------
    namespace test
    {
        //-----------------------------------------------------------------------------
        //! The test stream specifics.
        //-----------------------------------------------------------------------------
        namespace stream
        {
            namespace traits
            {
                //#############################################################################
                //! The stream type trait for the stream that should be used for the given accelerator.
                //#############################################################################
                template<
                    typename TDev,
                    typename TSfinae = void>
                struct DefaultStreamType
                {
#if (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                    using type = alpaka::stream::StreamCpuSync;
#else
                    using type = alpaka::stream::StreamCpuAsync;
#endif
                };

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__)
                //#############################################################################
                //! The stream type trait specialization for the CUDA accelerator.
                //#############################################################################
                template<>
                struct DefaultStreamType<
                    alpaka::dev::DevCudaRt>
                {
#if (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                    using type = alpaka::stream::StreamCudaRtSync;
#else
                    using type = alpaka::stream::StreamCudaRtAsync;
#endif
                };
#endif
            }
            //#############################################################################
            //! The stream type that should be used for the given accelerator.
            //#############################################################################
            template<
                typename TAcc>
            using DefaultStream = typename traits::DefaultStreamType<TAcc>::type;
        }
    }
}
