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

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
    #include <alpaka/accs/serial/Serial.hpp>
#endif
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
    #include <alpaka/accs/threads/Threads.hpp>
#endif
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED
    #include <alpaka/accs/fibers/Fibers.hpp>
#endif
#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
    #if _OPENMP < 200203
        #error If ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED is set, the compiler has to support OpenMP 2.0 or higher!
    #endif
    #include <alpaka/accs/omp/omp2/blocks/Omp2Blocks.hpp>
#endif
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
    #if _OPENMP < 200203
        #error If ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED is set, the compiler has to support OpenMP 2.0 or higher!
    #endif
    #include <alpaka/accs/omp/omp2/threads/Omp2Threads.hpp>
#endif
#ifdef ALPAKA_ACC_CPU_BT_OMP4_ENABLED
    #if _OPENMP < 201307
        #error If ALPAKA_ACC_CPU_BT_OMP4_ENABLED is set, the compiler has to support OpenMP 4.0 or higher!
    #endif
    #include <alpaka/accs/omp/omp4/cpu/Omp4Cpu.hpp>
#endif
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__)
    #include <alpaka/accs/cuda/Cuda.hpp>
#endif
