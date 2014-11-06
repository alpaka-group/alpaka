/**
* Copyright 2014 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of of either the GNU General Public License or
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

#include <alpaka/IAcc.hpp>

#ifdef ALPAKA_SERIAL_ENABLED
    #include <alpaka/AccSerial.hpp>
#endif
#ifdef ALPAKA_THREADS_ENABLED
    #include <alpaka/AccThreads.hpp>
#endif
#ifdef ALPAKA_FIBERS_ENABLED
    #include <alpaka/AccFibers.hpp>
#endif
#ifdef ALPAKA_OPENMP_ENABLED
    #include <alpaka/AccOpenMp.hpp>
#endif
#ifdef ALPAKA_CUDA_ENABLED
    #include <alpaka/AccCuda.hpp>
#endif

#include <ostream>                  // std::ostream

namespace alpaka
{
    //#############################################################################
    //! Builds a kernel executor.
    //#############################################################################
    template<typename TAcc, typename TKernel, typename TWorkSize>
    auto buildKernelExecutor(TWorkSize const & work = TWorkSize(), TKernel const & kernel = TKernel())
        -> typename std::result_of<detail::KernelExecutorBuilder<TAcc, TKernel, TWorkSize>(TWorkSize, TKernel)>::type
    {
        // Use the specialized KernelExecutorBuilder for the given accelerator.
        return detail::KernelExecutorBuilder<TAcc, TKernel, TWorkSize>()(work, kernel);
    }

    /*//#############################################################################
    //! The available accelerator environments.
    //#############################################################################
    enum class EAccelerator
    {
#ifdef ALPAKA_SERIAL_ENABLED
        Serial,
#endif
#ifdef ALPAKA_THREADS_ENABLED
        Threads,
#endif
#ifdef ALPAKA_FIBERS_ENABLED
        Fibers,
#endif
#ifdef ALPAKA_OPENMP_ENABLED
        OpenMp,
#endif
#ifdef ALPAKA_CUDA_ENABLED
        Cuda,
#endif
    };

    //-----------------------------------------------------------------------------
    //! Stream out the name of the accelerator environment.
    //-----------------------------------------------------------------------------
    std::ostream& operator << (std::ostream & os, EAccelerator const & eDevice)
    {
#ifdef ALPAKA_SERIAL_ENABLED
        if(eDevice == EAccelerator::Serial)
        {
            os << "Serial";
        }
        else
#endif
#ifdef ALPAKA_THREADS_ENABLED
        if(eDevice == EAccelerator::Threads)
        {
            os << "Threads";
        }
        else
#endif
#ifdef ALPAKA_FIBERS_ENABLED
        if(eDevice == EAccelerator::Fibers)
        {
            os << "Fibers";
        }
        else
#endif
#ifdef ALPAKA_OPENMP_ENABLED
        if(eDevice == EAccelerator::OpenMp)
        {
            os << "OpenMp";
        }
        else
#endif
#ifdef ALPAKA_CUDA_ENABLED
        if(eDevice == EAccelerator::Cuda)
        {
            os << "Cuda";
        }
        else
#endif 
        {
            os << "<unknown>";
        }
        return os;
    }*/

    //-----------------------------------------------------------------------------
    //! Builds a kernel executor.
    //-----------------------------------------------------------------------------
    /*template<typename TKernel, typename TWorkSize>
    auto buildKernelExecutor(EAccelerator const eDevice, TWorkSize work = TWorkSize())
        -> FIXME: What is the return type?
    {
#ifdef ALPAKA_SERIAL_ENABLED
        if(eDevice == EAccelerator::Serial)
        {
            buildKernelExecutor<AccSerial, TKernel>(work);
        }
        else
#endif
#ifdef ALPAKA_THREADS_ENABLED
        if(eDevice == EAccelerator::Threads)
        {
            buildKernelExecutor<AccThreads, TKernel>(work);
        }
        else
#endif
#ifdef ALPAKA_FIBERS_ENABLED
        if(eDevice == EAccelerator::Fibers)
        {
            buildKernelExecutor<AccFibers, TKernel>(work);
        }
        else
#endif
#ifdef ALPAKA_OPENMP_ENABLED
        if(eDevice == EAccelerator::OpenMp)
        {
            buildKernelExecutor<AccOpenMp, TKernel>(work);
        }
        else
#endif
#ifdef ALPAKA_CUDA_ENABLED
        if(eDevice == EAccelerator::Cuda)
        {
            buildKernelExecutor<AccCuda, TKernel>(work);
        }
        else
#endif 
        {
            std::cout << "<unknown accelerator>" << std::endl;
        }
    }*/
}