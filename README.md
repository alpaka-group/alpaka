**alpaka** - Abstraction Library for Parallel Kernel Acceleration
=================================================================

The **alpaka** library is a header-only C++11 abstraction library for accelerator development.

Its aim is to provide performance portability through the abstraction (not hiding!) of the underlying levels of parallelism.

It is platform independent and supports the concurrent and cooperative use of multiple devices such as CPUs, CUDA GPUs and Xeon Phis with a multitude of accelerator-backends such as CUDA, OpenMP (2.0), Boost.Fiber, std::thread and serial execution (where applicaple). Only one implementation of the kernel is required by utilizing the uniform kernel interface.

The **alpaka** API is currently unstable (alpha state).

The library allows users to utilize a multitude of different accelerator types that require different libraries/compilers by providing a uniform kernel interface.
Users have to write only one implementation of their algorithms and can benefit from all supported accelerators.
There is no need to write special CUDA, OpenMP or custom threading code.
The supported accelerators can be selected at compile time but the decision which accelerator executes which kernel can be made at runtime.

The abstraction used is very similar to the CUDA grid-blocks-threads division strategy.
Algorithms that should be parallelized have to be divided into a 1, 2, or 3-dimensional grid consisting of small uniform work items.
The function being executed by each of this threads is called a kernel. 
The threads in the grid are organized in blocks.
All threads in a block are executed in parallel and can interact via fast shared memory.
Blocks are executed independently and can not interact in any way.
The block execution order is unspecified and depends on the accelerator in use.
By using this abstraction the execution can be optimally adapted to the available accelerators.
 
Software License
----------------

**alpaka** is licensed under **LGPLv3** or later.


Documentation
-------------

The source code documentation generated with [doxygen](http://www.doxygen.org) is available [here](http://computationalradiationphysics.github.io/alpaka/).


Supported Compilers
-------------------

This library uses C++11 (or newer when available).

|-|gcc 4.9.2|gcc 5.1|clang 3.5+|MSVC 2013|MSVC 2015|icc 15.0+ (untested)|
|---|---|---|---|---|---|---|
|CUDA 7.0|:white_check_mark:|:x:|:x:|:white_check_mark:|:x:|:white_check_mark:|
|OpenMP 2.0|:white_check_mark:|:white_check_mark:|:x:|:white_check_mark:|:white_check_mark:|:white_check_mark:|
|Bopost.Fiber|:white_check_mark:|:white_check_mark:|:white_check_mark:|:x:|:x:|:white_check_mark:|
|std::thread|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|
|Serial|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|

**NOTE**: :bangbang: Currently the CUDA-Accelerator can not be enabled together with the Threads-Accelerator or Fibers-Accelerator :bangbang:

Build status master branch: [![Build Status](https://travis-ci.org/ComputationalRadiationPhysics/alpaka.svg?branch=master)](https://travis-ci.org/ComputationalRadiationPhysics/alpaka)

Build status develop branch: [![Build Status](https://travis-ci.org/ComputationalRadiationPhysics/alpaka.svg?branch=develop)](https://travis-ci.org/ComputationalRadiationPhysics/alpaka)


Accelerator Backends
------------

|-|Serial|std::thread|Boost.Fiber|OpenMP 2.0|CUDA 7.0|
|---|---|---|---|---|---|
|Devices|Host Core|Host Cores|Host Core|Host Cores|NVIDIA GPUs|
|Lib/API|n/a| std::thread | boost::fibers::fiber |OpenMP 2.0|CUDA 7.0|
|Execution strategy grid-blocks|sequential|sequential|sequential|sequential|undefined|
|Execution strategy block-threads|sequential|preemptive multitasking|cooperative multitasking|preemptive multitasking|lock-step within warps|


Dependencies
------------

[Boost](http://boost.org/) 1.56+ is the only mandatory external dependency.
Just header-only libraries are required by the **alpaka** library itself.
However some of the examples require different boost libraries to be built.

When the *CUDA-Accelerator* is enabled, version *7.0* of the *CUDA SDK* is the minimum requirement.

When the *OpenMP-Accelerator* is enabled, the compiler and the platform have to support *OpenMP 2.0* or newer.

When the *Fibers-Accelerator* is enabled, the develop branch of boost and the proposed boost library [`boost-fibers`](https://github.com/olk/boost-fiber) (develop branch) are required. `boost-fibers`, `boost-context` and all of its dependencies are required to be build in C++14 mode `./b2 cxxflags="-std=c++14"`.


Usage
-----

The library is header only so nothing has to be build.
Only the include path (`-I` or `export CPLUS_INCLUDE_PATH=`) has to be set to `<PATH-TO-ALPAKA-LIB>/include/`.
This allows to include the whole alpaka library with: `#include <alpaka/alpaka.hpp>`

Code not intended to be utilized by users is hidden in the `detail` namespace.

If you are building with the *CUDA-Accelerator* enabled, your source files are required to have the ending `.cu` to comply with the nvcc (NVIDIA CUDA C++ compiler) rules for code files using CUDA.
When the *CUDA-Accelerator* is disabled, this is not required and a `.cpp` extension is enough.
To allow both use-cases, it is desirable to have both, a `.cpp` file with the implementation and a `.cu` file containing only `#include <PATH/TO/IMPL.cpp>` to forward to the implementation.
The build system then has to use the `.cu` files when the *CUDA-Accelerator* is enabled and the `.cpp` files else.


Authors
-------

### Maintainers and core developers

- Benjamin Worpitz

### Scientific Supervision

- Dr. Michael Bussmann

### Participants, Former Members and Thanks

- Rene Widera
- Axel Huebl
