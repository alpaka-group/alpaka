alpaka - Abstraction Library for Parallel Kernel Acceleration
================================================================

The alpaka library allows users to utilize a multitude of different accelerators that require different libraries/compilers by providing a uniform kernel interface.
The abstraction used is very similar to the CUDA grid-blocks-threads division strategy.
Users have to write only one implementation of their algorithms and can benefit from all supported accelerators.
There is no need to write special CUDA, OpenMP or custom threading code.
The supported accelerators can be selected at compile time but the decision which accelerator executes which kernel can be made at runtime.

Software License
----------------

*alpaka* is licensed under the **GPLv3+**. 

Supported Compilers
-------------------

This library uses a subset of C++11 supported by many compilers to keep the
code clean and readable.

Supported (but not necessarily tested) compilers are:
- gcc 4.7+
- clang 3.4+
- icc 15.0+
- MSVC 2015+

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
|Lib/API|n/a|std::thread|boost::fibers::fiber|OpenMP 2.0|CUDA 6.5|
|Execution strategy grid-blocks|sequential|sequential|sequential|sequential|undefined|
|Execution strategy block-kernels|sequential|preemptive multitasking|cooperative multithreading|preemptive multitasking|lock-step within warps|

Requirements
------------

[Boost](http://boost.org/) 1.55+ is the only required external dependency.
By default only header-only libraries are used.

When the **Fibers-Accelerator** is enabled, `boost-coroutine`, `boost-context` and the proposed boost library [`boost-fibers`](https://github.com/olk/boost-fiber) (develop branch) are required to be build.

When the **CUDA-Accelerator** is enabled, version *6.5* of the *CUDA SDK* is the minimum requirement.

When the **OpenMP-Accelerator** is enabled, the compiler and the platform have to support *OpenMP 2.0* or newer.

Usage
-----

The library is header only so nothing has to be build.
Only the include path (`-I` or `export CPLUS_INCLUDE_PATH=`) has to be set to `<PATH-TO-LIB>/include/`.
This allows the usage of header inclusion in the following way:

```c++
#include <alpaka/alpaka.hpp>
```

Code not intended to be utilized by users is hidden in the `detail` namespace.
