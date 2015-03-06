alpaka - Abstraction Library for Parallel Kernel Acceleration
=============================================================

The alpaka library allows users to utilize a multitude of different accelerator types that require different libraries/compilers by providing a uniform kernel interface.
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

*alpaka* is licensed under the **LGPLv3+**.


Documentation
-------------

The source code documentation generated with [doxygen](www.doxygen.org) is available [here](http://computationalradiationphysics.github.io/alpaka/).


Supported Compilers
-------------------

This library uses a subset of C++11 supported by many compilers to keep the code clean and readable.

Supported compilers are:
- gcc 4.8.2+ (boost-fibers only supported in gcc 4.9+)
- MSVC 2013+ (boost-fibers only supported in MSVC 2015)
- clang 3.4+ (currently OpenMP only supported in `clang-omp` )
- icc 15.0+ (untested)

Build status master branch: [![Build Status](https://travis-ci.org/ComputationalRadiationPhysics/alpaka.svg?branch=master)](https://travis-ci.org/ComputationalRadiationPhysics/alpaka)

Build status develop branch: [![Build Status](https://travis-ci.org/ComputationalRadiationPhysics/alpaka.svg?branch=develop)](https://travis-ci.org/ComputationalRadiationPhysics/alpaka)


Requirements
------------

[Boost](http://boost.org/) 1.55+ is the only required external dependency.
By default just header-only libraries are used.

When the **Fibers-Accelerator** is enabled, `boost-coroutine`, `boost-context` and the proposed boost library [`boost-fibers`](https://github.com/olk/boost-fiber) (develop branch commit 6a1257442bb82e9082a55cacc2c6ebe02b4aa540) are required to be build.

When the **CUDA-Accelerator** is enabled, version *6.5* of the *CUDA SDK* is the minimum requirement.

When the **OpenMP-Accelerator** is enabled, the compiler and the platform have to support *OpenMP 2.0* or newer.


Accelerators
------------
- **Serial-Accelerator**
- **Threads-Accelerator**
- **Fibers-Accelerator**
- **OpenMP-Accelerator**
- **CUDA-Accelerator**

|-|serial|threads|fibers|OpenMP|CUDA|
|---|---|---|---|---|---|
|Devices|Host Core|Host Cores|Host Core|Host Cores|NVIDIA GPUs|
|Lib/API|n/a| std::thread | boost::fibers::fiber |OpenMP 2.0|CUDA 6.5|
|Execution strategy grid-blocks|sequential|sequential|sequential|sequential|undefined|
|Execution strategy block-threads|sequential|preemptive multitasking|cooperative multitasking|preemptive multitasking|lock-step within warps|


Usage
-----

The library is header only so nothing has to be build.
Only the include path (`-I` or `export CPLUS_INCLUDE_PATH=`) has to be set to `<PATH-TO-ALPAKA-LIB>/include/`.
This allows the usage of header inclusion in the following way:

```c++
#include <alpaka/alpaka.hpp>
```

Code not intended to be utilized by users is hidden in the `detail` namespace.

If you are building with the **CUDA-Accelerator** enabled, your source files are required to have the ending `.cu` to comply with the nvcc (NVIDIA CUDA C++ compiler) rules for code files using CUDA.
When the **CUDA-Accelerator** is disabled, this is not required and a `.cpp` extension is enough.
To allow both use-cases, it is desirable to have both, a `.cpp` file with the implementation and a `.cu` file containing only `#include <PATH/TO/IMPL.cpp>` to forward to the implementation.
The build system then has to use the `.cu` files when the **CUDA-Accelerator** is enabled and the `.cpp` files else.


Authors
-------

### Maintainers and core developers

- Benjamin Worpitz

### Scientific Supervision

- Dr. Michael Bussmann

### Participants, Former Members and Thanks

- Rene Widera
- Axel Huebl
