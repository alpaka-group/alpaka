Distinction
===========

There are multiple other projects which pretend to achieve full performance portability.
Many do not satisfy the requirement for full c++ support that is needed because of the usage of template meta-programming for method specialization to avoid runtime-polymorphism.


CUDA
------

### Positive
* Full control over memory, caches and execution.

### Negative
* Not platform independent: NVIDIA GPUs are required. No execution on other manufacturers GPUs or even standard CPUs.
* Language extension: The nvcc compiler driver is required to understand the language constructs that are used inside the kernel methods.
* Not compiler independent: The back-end compiler has to be one of the supported ones.


OpenMP
--------

### Negative
* No good way to control memory hierarchies and thread interaction (shared memory).
* Only compiler hints, no direct control over result.


OpenACC
---------

### Positive
* Can generate x86 and CUDA code from C++ code.

### Negative
* Compiler dependent (currently not supported by many compilers and the PGI compiler is not actively enough developed and does not have a good C++ support).
* Only compiler hints, no direct control over result.


OpenCl
--------

### Positive
* Hardware independent (CPUs and GPUs of nearly all vendors).

### Negative
* No full C++ support.
* Runtime compilation -> No direct inclusion into the source (syntax highlighting?, static analysis?, debugging?, templated kernels?)


C++ AMP
---------

### Positive
* Open specification
* Annotated C++ code can run on multiple accelerators.

### Negative
* Language extension
* Compiler dependent (currently not supported by many compilers)


PGI CUDA-X86
-------------
When run on x86-based systems without a GPU, PGI CUDA C applications use multiple cores and the streaming SIMD (Single Instruction Multiple Data) capabilities of Intel and AMD CPUs for parallelvectorized execution.
At run-time, CUDA C programs compiled for x86 executes each CUDA thread block using a single host core, eliminating synchronization where possible.

### Positive
* Lets you write standard CUDA code and execute it on x86.

### Negative
* Not actively developed.


LLVM backends (PTX, R600)
---------------------------

### Negative
* Those back-ends never got really mature and up-to-date.


KOKKOS
-------------
See [here](https://www.xsede.org/documents/271087/586927/Edwards-2013-XSCALE13-Kokkos.pdf)
[here](http://trilinos.org/oldsite/events/trilinos_user_group_2013/presentations/2013-11-TUG-Kokkos-Tutorial.pdf)
[here](http://on-demand.gputechconf.com/supercomputing/2013/presentation/SC3103_Towards-Performance-Portable-Applications-Kokkos.pdf)
and  [here](http://dx.doi.org/10.3233/SPR-2012-0343).
Source is available [here](https://github.com/trilinos/trilinos/tree/master/packages/kokkos).
The project is similar to *alpaka* in the way it abstracts the kernel as templated function object.
It provides parallel_for, parallel_reduce, etc. similar to thrust.

### Positive
* Offers buffer views with a neutral indexing scheme that maps to the underlying hardware (row/col-major, blocking, ...). 

### Negative
* License.
* The parameters are required to be given to the function object constructor coupling algorithm and data together.
* The implementation of accelerator methods (atomics, ...) is selected via macros defined by the nvcc compiler. So there is no way to select between different x86 implementations for different x86 accelerators. 


Phalanx
-----------
See [here](http://www.mgarland.org/files/papers/phalanx-sc12-preprint.pdf).
It is very similar to *alpaka* in the way it abstracts the accelerators.

### Positive
* C++ Interface provides CUDA, OpenMP, and GASNet back-ends

### Negative
* License.
* No official source repository available?


thrust (bulk)
-------------
*...


Intel TBB
---------
*...


Intel Cilk Plus
---------------
*...
