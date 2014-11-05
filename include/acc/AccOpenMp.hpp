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

#include <acc/KernelExecutorBuilder.hpp>    // KernelExecutorBuilder
#include <acc/WorkSize.hpp>                    // IWorkSize, WorkSizeDefault

#include <cstddef>                            // std::size_t
#include <cstdint>                            // unit8_t
#include <vector>                            // std::vector
#include <cassert>                            // assert
#include <stdexcept>                        // std::except
#include <string>                            // std::to_string
#ifdef _DEBUG
    #include <iostream>                        // std::cout
#endif

#include <omp.h>

namespace acc
{
    //#############################################################################
    //! The base class for all OpenMP accelerated kernels.
    //#############################################################################
    class AccOpenMp :
        public detail::IWorkSize<detail::WorkSizeDefault>
    {
        using TWorkSize = detail::IWorkSize<detail::WorkSizeDefault>;
    public:
        //-----------------------------------------------------------------------------
        //! Constructor.
        //-----------------------------------------------------------------------------
        ACC_FCT_CPU AccOpenMp() = default;

        //-----------------------------------------------------------------------------
        //! \return The maximum number of kernels in each dimension of a tile allowed.
        //-----------------------------------------------------------------------------
        ACC_FCT_CPU static vec<3> getSizeTileKernelsMax()
        {
            auto const uiSizeTileKernelsLinearMax(getSizeTileKernelsLinearMax());
            return {uiSizeTileKernelsLinearMax, uiSizeTileKernelsLinearMax, uiSizeTileKernelsLinearMax};
        }
        //-----------------------------------------------------------------------------
        //! \return The maximum number of kernels in a tile allowed.
        //-----------------------------------------------------------------------------
        ACC_FCT_CPU static std::uint32_t getSizeTileKernelsLinearMax()
        {
            // HACK: ::omp_get_max_threads() does not return the real limit of the underlying OpenMP runtime:
            // 'The omp_get_max_threads routine returns the value of the nthreads-var internal control variable, which is used to determine the number of threads that would form the new team, 
            // if an active parallel region without a num_threads clause were to be encountered at that point in the program.'
            // How to do this correctly? Is there even a way to get the hard limit apart from omp_set_num_threads(high_value) -> omp_get_max_threads()?
            ::omp_set_num_threads(1024);
            return ::omp_get_max_threads();
        }

    protected:
        //-----------------------------------------------------------------------------
        //! \return The thread index of the currently executed kernel.
        //-----------------------------------------------------------------------------
        ACC_FCT_CPU vec<3> getIdxTileKernel() const
        {
            vec<3> v3uiIdxTileKernel;

            auto const v3uiSizeTileKernels(TWorkSize::getSize<Tile, Kernels, D3>());
            auto const t(::omp_get_thread_num());
            v3uiIdxTileKernel[0] = (t % (v3uiSizeTileKernels[1] * v3uiSizeTileKernels[0])) % v3uiSizeTileKernels[0];
            v3uiIdxTileKernel[1] = (t % (v3uiSizeTileKernels[1] * v3uiSizeTileKernels[0])) / v3uiSizeTileKernels[0];
            v3uiIdxTileKernel[2] = (t / (v3uiSizeTileKernels[1] * v3uiSizeTileKernels[0]));

            return v3uiIdxTileKernel;
        }
        //-----------------------------------------------------------------------------
        //! \return The block index of the currently executed kernel.
        //-----------------------------------------------------------------------------
        ACC_FCT_CPU vec<3> getIdxGridTile() const
        {
            return m_v3uiGridTileIdx;
        }

        //-----------------------------------------------------------------------------
        //! Atomic addition.
        //-----------------------------------------------------------------------------
        template<typename T>
        ACC_FCT_CPU void atomicFetchAdd(T * sum, T summand) const
        {
            auto & rsum(*sum);
            // NOTE: Braces or calling other functions directly after 'atomic' are not allowed!
            #pragma omp atomic
            rsum += summand;
        }

        //-----------------------------------------------------------------------------
        //! Syncs all threads in the current block.
        //-----------------------------------------------------------------------------
        ACC_FCT_CPU void syncTileKernels() const
        {
            #pragma omp barrier
        }

        //-----------------------------------------------------------------------------
        //! \return The pointer to the block shared memory.
        //-----------------------------------------------------------------------------
        template<typename T>
        ACC_FCT_CPU T * getTileSharedExternMem() const
        {
            return reinterpret_cast<T*>(m_vuiSharedMem.data());
        }

    private:
        mutable std::vector<uint8_t> m_vuiSharedMem;    //!< Block shared memory.

        mutable vec<3> m_v3uiGridTileIdx;                //!< The index of the currently executed block.

    public:
        //#############################################################################
        //! The executor for an accelerated serial kernel.
        //#############################################################################
        template<typename TAccedKernel>
        class KernelExecutor :
            protected TAccedKernel
        {
        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            template<typename TWorkSize>
            KernelExecutor(TWorkSize workSize)
            {
                (*static_cast<typename AccOpenMp::TWorkSize*>(this)) = workSize;
#ifdef _DEBUG
                std::cout << "AccOpenMp::KernelExecutor()" << std::endl;
#endif
            }

            //-----------------------------------------------------------------------------
            //! Executes the accelerated kernel.
            //-----------------------------------------------------------------------------
            template<typename... TArgs>
            void operator()(TArgs && ... args) const
            {
#ifdef _DEBUG
                std::cout << "[+] AccOpenMp::KernelExecutor::operator()" << std::endl;
#endif

                auto const uiNumKernelsPerTile(this->AccOpenMp::template getSize<Tile, Kernels, Linear>());
                auto const uiMaxKernelsPerTile(AccOpenMp::getSizeTileKernelsLinearMax());
                if(uiNumKernelsPerTile > uiMaxKernelsPerTile)
                {
                    throw std::runtime_error(("The given tileSize '" + std::to_string(uiNumKernelsPerTile) + "' is larger then the supported maximum of '" + std::to_string(uiMaxKernelsPerTile) + "' by the OpenMp accelerator!").c_str());
                }

                auto const v3uiSizeGridTiles(this->AccOpenMp::template getSize<Grid, Tiles, D3>());
                auto const v3uiSizeTileKernels(this->AccOpenMp::template getSize<Tile, Kernels, D3>());
#ifdef _DEBUG
                //std::cout << "GridTiles: " << v3uiSizeGridTiles << " TileKernels: " << v3uiSizeTileKernels << std::endl;
#endif

                this->AccOpenMp::m_vuiSharedMem.resize(TAccedKernel::getBlockSharedMemSizeBytes(v3uiSizeTileKernels));

                // CUDA programming guide: "Thread blocks are required to execute independently: It must be possible to execute them in any order, in parallel or in series. 
                // This independence requirement allows thread blocks to be scheduled in any order across any number of cores"
                // -> We can execute them serially.
                for(std::uint32_t bz(0); bz<v3uiSizeGridTiles[2]; ++bz)
                {
                    this->AccOpenMp::m_v3uiGridTileIdx[2] = bz;
                    for(std::uint32_t by(0); by<v3uiSizeGridTiles[1]; ++by)
                    {
                        this->AccOpenMp::m_v3uiGridTileIdx[1] = by;
                        for(std::uint32_t bx(0); bx<v3uiSizeGridTiles[0]; ++bx)
                        {
                            this->AccOpenMp::m_v3uiGridTileIdx[0] = bx;

                            // The number of threads in this block.
                            std::uint32_t const uiNumKernelsInTile(this->TAccedKernel::template getSize<Tile, Kernels, Linear>());

                            // Force the environment to use the given number of threads.
                            ::omp_set_dynamic(0);

                            // Parallelizing the threads is required because when syncTileKernels is called all of them have to be done with their work up to this line.
                            // So we have to spawn one real thread per thread in a block.
                            // 'omp for' is not useful because it is meant for cases where multiple iterations are executed by one thread but in our cas a 1:1 mapping is required.
                            // Therefore we use 'omp parallel' with the specified number of threads in a block.
                            // FIXME: Does this hinder executing multiple kernels in parallel because their block sizes/omp thread numbers are interfering? Is this a real use case? 
                            #pragma omp parallel num_threads(uiNumKernelsInTile)
                            {
#ifdef _DEBUG
                                if((::omp_get_thread_num() == 0) && (bz == 0) && (by == 0) && (bx == 0))
                                {
                                    std::cout << "omp_get_num_threads: " << ::omp_get_num_threads() << std::endl;
                                }
#endif
                                this->TAccedKernel::operator()(std::forward<TArgs>(args)...);

                                // Wait for all threads to finish before deleting the shared memory.
                                this->AccOpenMp::syncTileKernels();
                            }
                        }
                    }
                }

                // After all blocks have been processed, the shared memory can be deleted.
                this->AccOpenMp::m_vuiSharedMem.clear();
#ifdef _DEBUG
                std::cout << "[-] AccOpenMp::KernelExecutor::operator()" << std::endl;
#endif
            }
        };
    };

    namespace detail
    {
        //#############################################################################
        //! The serial kernel executor builder.
        //#############################################################################
        template<template<typename> class TKernel, typename TWorkSize>
        class KernelExecutorBuilder<AccOpenMp, TKernel, TWorkSize>
        {
        public:
            using TAccedKernel = TKernel<IAcc<AccOpenMp>>;
            using TKernelExecutor = AccOpenMp::KernelExecutor<TAccedKernel>;

            //-----------------------------------------------------------------------------
            //! Creates an kernel executor for the serial accelerator.
            //-----------------------------------------------------------------------------
            TKernelExecutor operator()(TWorkSize workSize) const
            {
                return TKernelExecutor(workSize);
            }
        };
    }
}