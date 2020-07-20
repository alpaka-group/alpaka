Similar Projects
================

There are multiple other libraries targeting the (portable) parallel task execution within nodes.
Some of them require language extensions, others pretend to achieve full performance portability across a multitude of devices.
But none of these libraries can provide full control over the (possibly diverse) underlying hardware while being only minimal invasive.
There is always a productivity-performance trade-off.

Furthermore, many of the libraries do not satisfy the requirement for full single-source C++ support.
This is essential because many simulation codes heavily rely on template meta-programming for method specialization and compile time optimizations.

`KOKKOS <https://github.com/kokkos>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. seealso::
   * https://www.xsede.org/documents/271087/586927/Edwards-2013-XSCALE13-Kokkos.pdf
   * https://trilinos.org/oldsite/events/trilinos_user_group_2013/presentations/2013-11-TUG-Kokkos-Tutorial.pdf
   * https://on-demand.gputechconf.com/supercomputing/2013/presentation/SC3103\_Towards-Performance-Portable-Applications-Kokkos.pdf
   * https://dx.doi.org/10.3233/SPR-2012-0343

provides an abstract interface for portable, performant shared memory-programming.
It is a C++ library that offers ``parallel_for``, ``parallel_reduce`` and similar functions for describing the pattern of the parallel tasks.
The execution policy determines how the threads are executed.
For example, this influences the sizes of blocks of threads or if static or dynamic scheduling should be used.
The library abstracts the kernel as a function object that can not have any user defined parameters for its ``operator()``.
Inconveniently, arguments have to be stored in members of the function object coupling algorithm and data together.
*KOKKOS* provides both, abstractions for parallel execution of code and data management.
Multidimensional arrays with a neutral indexing and an architecture dependent layout are available, which can be used, for example, to abstract the underlying hardwares preferred memory access scheme that could be row-major, column-major or even blocked.


`Thrust <https://thrust.github.io/>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

is a parallel algorithms library resembling the C++ Standard Template Library (STL).
It allows to select either the *CUDA*, *TBB* or *OpenMP* back-end at make-time.
Because it is based on generic ``host_vector`` and ``device_vector`` container objects, it is tightly coupling the data structure and the parallelization strategy.
There exist many similar libraries such as `ArrayFire <https://arrayfire.com/>`_ (*CUDA*, *OpenCL*, native C++), `VexCL <https://github.com/ddemidov/vexcl/>`_ (*OpenCL*, *CUDA*), `ViennaCL <http://viennacl.sourceforge.net/>`_ (*OpenCL*, *CUDA*, *OpenMP*) and `hemi <https://github.com/harrism/hemi/>`_ (*CUDA*, native C++).

.. seealso::
   * Phalanx
     See `here <https://www.mgarland.org/files/papers/phalanx-sc12-preprint.pdf>`_
     It is very similar to *alpaka* in the way it abstracts the accelerators.
     C++ Interface provides CUDA, OpenMP, and GASNet back-ends
   * Aura
   * Intel TBB
   * U\PC++
