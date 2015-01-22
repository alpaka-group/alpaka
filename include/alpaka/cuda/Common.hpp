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

#include <iostream>		// std::cerr

#include <stdexcept>	// std::runtime_error

#include <cuda.h>       // cudaGetErrorString

#if (!defined(CUDA_VERSION) || (CUDA_VERSION < 6050))
	#error "CUDA version 6.5 or greater required!"
#endif

 //! Error checking log only.
#define ALPAKA_CUDA_CHECK_MSG(cmd, msg)\
	{\
        cudaError_t error = cmd;\
        if(error != cudaSuccess)\
        {\
		    std::cerr << "<" << __FILE__ << ">:" << __LINE__ << msg << std::endl;\
        }\
	}

//! Error checking with log and exception.
#define ALPAKA_CUDA_CHECK_MSG_EXCP(cmd)\
	{\
        cudaError_t error = cmd;\
        if(error != cudaSuccess)\
	    {\
		     std::cerr << "<" << __FILE__ << ">:" << __LINE__ << std::endl;\
		     throw std::runtime_error(std::string("[CUDA] Error: ") + std::string(cudaGetErrorString(error)));\
        }\
	}

//! The default error checking.
#define ALPAKA_CUDA_CHECK(cmd) ALPAKA_CUDA_CHECK_MSG_EXCP(cmd)

