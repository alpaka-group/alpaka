acc - A Abstract Many-Core Acceleration Library 
================================================================

Software License
----------------

*PIConGPU* is licensed under the **GPLv3+**. 

Supported Compilers
----------------

This library uses a subset of C++11 supported by many compilers to keep the code clean and readable.
This compiles with:
- gcc 4.4+
- clang 3.3+
- icc 13.0+
- VC12+

When using the CUDA-Accelerator back-end, version 6.5 is the minimum requirement.

Usage
----------------

The library is header only so nothing has to be build. Only the include path has to be set to 'PATH/TO/LIB/include/'.

Code not intended to be utilized by users is hidden in the 'detail' namespace.