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

#ifdef ALPAKA_SERIAL_ENABLED
    #include <alpaka/serial/AccSerial.hpp>
#endif
#ifdef ALPAKA_THREADS_ENABLED
    #include <alpaka/threads/AccThreads.hpp>
#endif
#ifdef ALPAKA_FIBERS_ENABLED
    #include <alpaka/fibers/AccFibers.hpp>
#endif
#ifdef ALPAKA_OPENMP_ENABLED
    #include <alpaka/openmp/AccOpenMp.hpp>
#endif
#ifdef ALPAKA_CUDA_ENABLED
    #include <alpaka/cuda/AccCuda.hpp>
#endif

#include <alpaka/interfaces/Event.hpp>
#include <alpaka/interfaces/IAcc.hpp>
#include <alpaka/interfaces/KernelExecCreator.hpp>
#include <alpaka/interfaces/BlockSharedExternMemSizeBytes.hpp>
#include <alpaka/interfaces/Memory.hpp>

#include <alpaka/host/WorkSize.hpp>         // alpaka::WorkSizeHost

#include <iostream>                         // std::cout

//-----------------------------------------------------------------------------
//! The alpaka library namespace.
//-----------------------------------------------------------------------------
namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! Logs the enabled accelerators.
    //-----------------------------------------------------------------------------
    static void logEnabledAccelerators()
    {
        std::cout << "Accelerators enabled: ";
#ifdef ALPAKA_SERIAL_ENABLED
        std::cout << "ALPAKA_SERIAL_ENABLED ";
#endif
#ifdef ALPAKA_THREADS_ENABLED
        std::cout << "ALPAKA_THREADS_ENABLED ";
#endif
#ifdef ALPAKA_FIBERS_ENABLED
        std::cout << "ALPAKA_FIBERS_ENABLED ";
#endif
#ifdef ALPAKA_OPENMP_ENABLED
        std::cout << "ALPAKA_OPENMP_ENABLED ";
#endif
#ifdef ALPAKA_CUDA_ENABLED
        std::cout << "ALPAKA_CUDA_ENABLED ";
#endif
        std::cout << std::endl;
    }
}
