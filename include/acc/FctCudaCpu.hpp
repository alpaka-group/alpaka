/**
* Copyright 2014 Benjamin Worpitz
*
* This file is part of acc.
*
* acc is free software: you can redistribute it and/or modify
* it under the terms of of either the GNU General Public License or
* the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* acc is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License and the GNU Lesser General Public License
* for more details.
*
* You should have received a copy of the GNU General Public License
* and the GNU Lesser General Public License along with acc.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

// Do not use __device__ or __host__ attributes if we are not compiling with nvcc.
#if defined ACC_CUDA_ENABLED && defined __CUDACC__
    #define ACC_FCT_CUDA __device__ __forceinline__
    #define ACC_FCT_CPU_CUDA __device__ __host__ __forceinline__
    #define ACC_FCT_CPU __host__ inline
#else
    #define ACC_FCT_CUDA inline
    #define ACC_FCT_CPU_CUDA inline
    #define ACC_FCT_CPU inline
#endif