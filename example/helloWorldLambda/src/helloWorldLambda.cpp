/**
 * \file
 * Copyright 2014-2015 Erik Zenker
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
 *
 */

// Alpaka
#include <alpaka/alpaka.hpp>

// STL
#include <iostream>
#include <functional>

/**
 * This functions says hi to the world and
 * can be encapsulated into a std::function
 * and used as a kernel function. It is 
 * just another way to define alpaka kernels
 * and might be useful when it is necessary
 * to lift an existing function into a kernel
 * function.
 */
template<typename Acc>
void hiWorldFunction(Acc& acc, size_t const nExclamationMarks){
    auto globalThreadIdx    = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
    auto globalThreadExtent = alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
    auto linearizedGlobalThreadIdx = alpaka::core::mapIdx<1u>(globalThreadIdx,
                                                              globalThreadExtent);
                                                          
    printf("[z:%u, y:%u, x:%u][linear:%u] Hi world from a std::function",
           static_cast<unsigned>(globalThreadIdx[0]),
           static_cast<unsigned>(globalThreadIdx[1]),
           static_cast<unsigned>(globalThreadIdx[2]),
           static_cast<unsigned>(linearizedGlobalThreadIdx[0]));

    for(size_t i = 0; i < nExclamationMarks; ++i){
        printf("!");
    }
                                                          
    printf("\n");  
}

int main() {

    /***************************************************************************
     * Define accelerator types
     **************************************************************************/
    using Dim     = alpaka::dim::DimInt<3>;
    using Size    = std::size_t;
    using Acc     = alpaka::acc::AccCpuSerial<Dim, Size>;
    using Stream  = alpaka::stream::StreamCpuSync;
    using DevAcc  = alpaka::dev::Dev<Acc>;
    using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Size>;

    /**************************************************************************
     * Get the first devices
     **************************************************************************/
    DevAcc  devAcc  (alpaka::dev::DevMan<Acc>::getDevByIdx(0));

    /***************************************************************************
     * Create a stream to the accelerator device
     **************************************************************************/
    Stream  stream  (devAcc);

    /***************************************************************************
     * Init workdiv
     **************************************************************************/
    alpaka::Vec<Dim, Size> const elementsPerThread(static_cast<Size>(1),
                                                   static_cast<Size>(1),
                                                   static_cast<Size>(1));

    alpaka::Vec<Dim, Size> const threadsPerBlock(static_cast<Size>(1),
                                                 static_cast<Size>(1),
                                                 static_cast<Size>(1));

    alpaka::Vec<Dim, Size> const blocksPerGrid(static_cast<Size>(2),
                                               static_cast<Size>(4),
                                               static_cast<Size>(8));

    WorkDiv const workdiv(alpaka::workdiv::WorkDivMembers<Dim, Size>(blocksPerGrid,
                                                                     threadsPerBlock,
                                                                     elementsPerThread));

    /**
     * Run kernel with lambda function
     *
     * Next to function objects which provide the
     * kernel function by overwrite the
     * operator(), alpaka is able to execute also
     * lambda functions (anonymous functions) which
     * are available since the C++11 standard.
     * Alpaka forces the lambda function to accept
     * the utilized accelerator as first argument. 
     * All following arguments can be provided after
     * the lambda function declaration. This example
     * passes the number exclamation marks, that should
     * be written after we greet the world, to the 
     * lambda function. This kind of kernel function
     * declaration might be useful when small kernels
     * are written for testing or lambda functions
     * allready exist.
     *
     */
    const size_t nExclamationMarks = 10;
    
    auto const helloWorld (alpaka::exec::create<Acc> (workdiv,
                                                      [](Acc& acc, size_t const nExclamationMarks) -> void {
                                                          auto globalThreadIdx    = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
                                                          auto globalThreadExtent = alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
                                                          auto linearizedGlobalThreadIdx = alpaka::core::mapIdx<1u>(globalThreadIdx,
                                                                                                                    globalThreadExtent);
                                                          
                                                          printf("[z:%u, y:%u, x:%u][linear:%u] Hello world from a lambda",
                                                                 static_cast<unsigned>(globalThreadIdx[0]),
                                                                 static_cast<unsigned>(globalThreadIdx[1]),
                                                                 static_cast<unsigned>(globalThreadIdx[2]),
                                                                 static_cast<unsigned>(linearizedGlobalThreadIdx[0]));

                                                          for(size_t i = 0; i < nExclamationMarks; ++i){
                                                              printf("!");
                                                          }
                                                          
                                                          printf("\n");

                                                      },
                                                      nExclamationMarks // 1st real kernel argument, but 2nd argument in the lambda function,
                                                                        // since, the first argument as allways the accelerator!
                                                      ));

    alpaka::stream::enqueue(stream, helloWorld);

    
    /**
     * Run kernel with std::function
     *
     * This kernel says hi to world by using 
     * std::functions, which are available since
     * the C++11 standard.
     * The interface for std::function can be used
     * to encapsulate normal c++ functions and 
     * lambda functions into a function object. 
     * Alpaka accepts these std::functions 
     * as kernel functions. Therefore, it is easy
     * to wrap allready existing code into a
     * std::function and provide it to the alpaka 
     * library.
     */
    auto const hiWorld (alpaka::exec::create<Acc> (workdiv,
                                                   std::function<void(Acc&, size_t)>( hiWorldFunction<Acc> ),
                                                   nExclamationMarks)); // 1st real kernel argument, but 2nd argument in the wrapped function
                                                                        // since, the first argument as allways the accelerator!
    alpaka::stream::enqueue(stream, hiWorld);    


    /**
     * Lambda function and std::function are neat, so lets return :)
     */
    return 0;
}
