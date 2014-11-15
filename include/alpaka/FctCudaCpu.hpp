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

// Do not use __device__ or __host__ attributes if we are not compiling with nvcc.
#if defined ALPAKA_CUDA_ENABLED && defined __CUDALPAKA__
    #define ALPAKA_FCT_CUDA __device__ __forceinline__
    #define ALPAKA_FCT_CPU_CUDA __device__ __host__ __forceinline__
    #define ALPAKA_FCT_CPU __host__ inline
#else
    #define ALPAKA_FCT_CUDA inline
    #define ALPAKA_FCT_CPU_CUDA inline
    #define ALPAKA_FCT_CPU inline
#endif