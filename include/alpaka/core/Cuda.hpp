/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 * SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-FileContributor: Sergei Bastrakov <s.bastrakov@hzdr.de>
 * SPDX-FileContributor: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
 * SPDX-FileContributor: Axel Hübl <a.huebl@plasma.ninja>
 * SPDX-FileContributor: Matthias Werner <Matthias.Werner1@tu-dresden.de>
 * SPDX-FileContributor: René Widera <r.widera@hzdr.de>
 * SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/CudaHipCommon.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

namespace alpaka::cuda::detail
{
    //! CUDA driver API error checking with log and exception, ignoring specific error values
    ALPAKA_FN_HOST inline auto cudaDrvCheck(CUresult const& error, char const* desc, char const* file, int const& line)
        -> void
    {
        if(error == CUDA_SUCCESS)
            return;

        char const* cu_err_name = nullptr;
        char const* cu_err_string = nullptr;
        CUresult cu_result_name = cuGetErrorName(error, &cu_err_name);
        CUresult cu_result_string = cuGetErrorString(error, &cu_err_string);
        std::string sError = std::string(file) + "(" + std::to_string(line) + ") " + std::string(desc) + " : '";
        if(cu_result_name == CUDA_SUCCESS && cu_result_string == CUDA_SUCCESS)
        {
            sError += std::string(cu_err_name) + "': '" + std::string(cu_err_string) + "'!";
        }
        else
        {
            // cuGetError*() failed, so append corresponding error message
            if(cu_result_name == CUDA_ERROR_INVALID_VALUE)
            {
                sError += " cuGetErrorName: 'Invalid Value'!";
            }
            if(cu_result_string == CUDA_ERROR_INVALID_VALUE)
            {
                sError += " cuGetErrorString: 'Invalid Value'!";
            }
        }
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
        std::cerr << sError << std::endl;
#    endif
        ALPAKA_DEBUG_BREAK;
        throw std::runtime_error(sError);
    }
} // namespace alpaka::cuda::detail

//! CUDA driver error checking with log and exception.
#    define ALPAKA_CUDA_DRV_CHECK(cmd) ::alpaka::cuda::detail::cudaDrvCheck(cmd, #cmd, __FILE__, __LINE__)

#    include "alpaka/core/UniformCudaHip.hpp"

#endif
