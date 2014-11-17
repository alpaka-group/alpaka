alpaka - Abstraction Library for Parallel Kernel Acceleration
================================================================

Software License
----------------

*alpaka* is licensed under the **GPLv3+**. 

Supported Compilers
----------------

This library uses a subset of C++11 supported by many compilers to keep the code clean and readable.
Supported (but not necessarily tested) compilers are:
- gcc 4.4+
- clang 3.3+
- icc 13.0+
- VC12+

Requirements
----------------

Boost is the only required external dependency. By default only header-only libraries are used.

When using the Fibers-Accelerator back-end, boost-coroutine, boost-context and the proposed boost library boost-fibers are required to be build.

When using the CUDA-Accelerator back-end, version 6.5 of the CUDA SDK is the minimum requirement.

When using the OpenMP-Accelerator back-end, the compiler and the platform have to support OpenMP 2.0 or newer.

Usage
----------------

The library is header only so nothing has to be build. Only the include path has to be set to 'PATH/TO/LIB/include/'.
This allows the usage of header inclusion in the following way: '#include &lt;alpaka/alpaka.hpp&gt;'.

Code not intended to be utilized by users is hidden in the 'detail' namespace.