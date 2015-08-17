**alpaka** - Abstraction Library for Parallel Kernel Acceleration
=================================================================

The **alpaka** library is a header-only C++11 abstraction library for accelerator development.

Its aim is to provide performance portability across accelerators through the abstraction (not hiding!) of the underlying levels of parallelism.

It is platform independent and supports the concurrent and cooperative use of multiple devices such as the hosts CPU as well as attached accelerators as for instance CUDA GPUs and Xeon Phis (currently native execution only).
A multitude of accelerator back-end variants using CUDA, OpenMP (2.0/4.0), Boost.Fiber, std::thread and also serial execution is provided and can be selected depending on the device.
Only one implementation of the user kernel is required by representing them as function objects with a special interface.
There is no need to write special CUDA, OpenMP or custom threading code.
Accelerator back-ends can be mixed within a device stream.
The decision which accelerator back-end executes which kernel can be made at runtime.

The **alpaka** API is currently unstable (alpha state).

The abstraction used is very similar to the CUDA grid-blocks-threads division strategy.
Algorithms that should be parallelized have to be divided into a multi-dimensional grid consisting of small uniform work items.
These functions are called kernels and are executed in parallel threads.
The threads in the grid are organized in blocks.
All threads in a block are executed in parallel and can interact via fast shared memory.
Blocks are executed independently and can not interact in any way.
The block execution order is unspecified and depends on the accelerator in use.
By using this abstraction the execution can be optimally adapted to the available hardware.


Software License
----------------

**alpaka** is licensed under **LGPLv3** or later.


Documentation
-------------

The source code documentation generated with [doxygen](http://www.doxygen.org) is available [here](http://computationalradiationphysics.github.io/alpaka/).


Accelerator Back-ends
---------------------

|Accelerator Back-end|Lib/API|Devices|Execution strategy grid-blocks|Execution strategy block-threads|
|---|---|---|---|---|
|Serial|n/a|Host CPU (single core)|sequential|sequential (only 1 thread per block)|
|OpenMP 2.0 blocks|OpenMP 2.0|Host CPU (multi core)|parallel (preemptive multitasking)|sequential (only 1 thread per block)|
|OpenMP 2.0 threads|OpenMP 2.0|Host CPU (multi core)|sequential|parallel (preemptive multitasking)|
|OpenMP 4.0|OpenMP 4.0|Host CPU (multi core)|parallel (undefined)|parallel (preemptive multitasking)|
| std::thread | std::thread |Host CPU (multi core)|sequential|parallel (preemptive multitasking)|
| Boost.Fiber | boost::fibers::fiber |Host CPU (single core)|sequential|parallel (cooperative multitasking)|
|CUDA 7.0|CUDA 7.0|NVIDIA GPUs SM 2.0+|parallel (undefined)|parallel (lock-step within warps)|


Supported Compilers
-------------------

This library uses C++11 (or newer when available).

|Accelerator Back-end|gcc 4.9.2|gcc 5.2|clang 3.5/3.6|clang 3.7|MSVC 2015|icc 15.0+ (untested)|
|---|---|---|---|---|---|---|
|Serial|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|
|OpenMP 2.0 blocks|:white_check_mark:|:white_check_mark:|:x:|:white_check_mark:|:white_check_mark:|:white_check_mark:|
|OpenMP 2.0 threads|:white_check_mark:|:white_check_mark:|:x:|:white_check_mark:|:white_check_mark:|:white_check_mark:|
|OpenMP 4.0|:white_check_mark:|:white_check_mark:|:x:|:x:|:x:|:white_check_mark:|
| std::thread |:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|
| Boost.Fiber |:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|
|CUDA 7.0|:white_check_mark:|:x:|:x:|:x:|:x:|:white_check_mark:|

**NOTE**: :bangbang: Currently the *CUDA accelerator back-end* can not be enabled together with the *std::thread accelerator back-end* or the *Boost.Fiber accelerator back-end* due to bugs in the NVIDIA nvcc compiler :bangbang:

Build status master branch: [![Build Status](https://travis-ci.org/ComputationalRadiationPhysics/alpaka.svg?branch=master)](https://travis-ci.org/ComputationalRadiationPhysics/alpaka)

Build status develop branch: [![Build Status](https://travis-ci.org/ComputationalRadiationPhysics/alpaka.svg?branch=develop)](https://travis-ci.org/ComputationalRadiationPhysics/alpaka)


Dependencies
------------

[Boost](http://boost.org/) 1.56+ is the only mandatory external dependency.
The **alpaka** library itself just requires header-only libraries.
However some of the accelerator back-end implementations require different boost libraries to be built.

When an accelerator back-end using *Boost.Fiber* is enabled, the develop branch of boost and the proposed boost library [`boost-fibers`](https://github.com/olk/boost-fiber) (develop branch) are required.
`boost-fibers`, `boost-context` and all of its dependencies are required to be build in C++14 mode `./b2 cxxflags="-std=c++14"`.

When an accelerator back-end using *CUDA* is enabled, version *7.0* of the *CUDA SDK* is the minimum requirement.

When an accelerator back-end using *OpenMP 2.0/4.0* is enabled, the compiler and the platform have to support the corresponding *OpenMP* version.


Usage
-----

The library is header only so nothing has to be build.
Only the include path (`-I` or `export CPLUS_INCLUDE_PATH=`) has to be set to `<PATH-TO-ALPAKA-LIB>/include/`.
This allows to include the whole alpaka library with: `#include <alpaka/alpaka.hpp>`

Code not intended to be utilized by users is hidden in the `detail` namespace.

If you are building with the *CUDA accelerator back-end* enabled, your source files are required to have the ending `.cu` to comply with the nvcc (NVIDIA CUDA C++ compiler) rules for code files using CUDA.
When the *CUDA accelerator back-end* is disabled, this is not required and a `.cpp` extension is enough.
To allow both use-cases, it is desirable to have both, a `.cpp` file with the implementation and a `.cu` file containing only `#include <PATH/TO/IMPL.cpp>` to forward to the implementation.
The build system then has to use the `.cu` files when the *CUDA accelerator back-end* is enabled and the `.cpp` files else.
Examples how to do this with CMake can be found in the `examples` folder.


Authors
-------

### Maintainers and core developers

- Benjamin Worpitz

### Participants, Former Members and Thanks

- Dr. Michael Bussmann
- Rene Widera
- Axel Huebl
