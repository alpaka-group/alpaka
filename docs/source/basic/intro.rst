Introduction
============

The *alpaka* library defines and implements an abstract interface for the *hierarchical redundant parallelism* model.
This model exploits task- and data-parallelism as well as memory hierarchies at all levels of current multi-core architectures.
This allows to achieve portability of performant codes across various types of accelerators by ignoring specific unsupported levels and utilizing only the ones supported on a specific accelerator.
All hardware types (multi- and many-core CPUs, GPUs and other accelerators) are treated and can be programmed in the same way.
The *alpaka* library provides back-ends for *CUDA*, *OpenMP*, *Boost.Fiber* and other methods.
The policy-based C++ template interface provided allows for straightforward user-defined extension of the library to support other accelerators.

The library name *alpaka* is an acronym standing for **A**\ bstraction **L**\ ibrary for **Pa**\ rallel **K**\ ernel **A**\ cceleration.

Distinction of the *alpaka* Library
-----------------------------------

In the section about the problems we saw that all portability problems of current HPC codes could be solved with an abstract interface unifying the underlying accelerator back-ends.
The previous section showed that there is currently no project available that could solve all of the problems highlighted.
The C++ interface library proposed to solve all those problems is called *alpaka*.
The subsequent enumeration will summarize the purpose of the library:

*alpaka* is ...
~~~~~~~~~~~~~~~

* an **abstract interface** describing parallel execution on multiple hierarchy levels. It allows to implement a mapping to various hardware architectures but **is no optimal mapping itself**.

* sustainably solving portability (50% on the way to reach full performance portability)

* solving the **heterogeneity** problem. An identical algorithm / kernel can be executed on heterogeneous parallel systems by selecting the target device.

* reducing the **maintainability** burden by not requiring to duplicate all the parts of the simulation that are directly facing the parallelization framework. Instead, it allows to provide a single version of the algorithm / kernel that can be used by all back-ends. All the accelerator dependent implementation details are hidden within the *alpaka* library.

* simplifying the **testability** by enabling **easy back-end switching**. No special hardware is required for testing the kernels. Even if the simulation itself will always use the *CUDA* back-end, the tests can completely run on a CPU. As long as the *alpaka* library is thoroughly tested for compatibility between the acceleration back-ends, the user simulation code is guaranteed to generate identical results (ignoring rounding errors / non-determinism) and is portable without any changes.

* **optimizable**. Everything in *alpaka* can be replaced by user code to optimize for special use-cases.

* **extensible**. Every concept described by the *alpaka* abstraction can be implemented by users. Therefore it is possible to non-intrusively define new devices, queues, buffer types or even whole accelerator back-ends.

* **data structure agnostic**. The user can use and define arbitrary data structures.

*alpaka* is not ...
~~~~~~~~~~~~~~~~~~~

* an automatically **optimal mapping** of algorithms / kernels to various acceleration platforms. Except in trivial examples an optimal execution always depends on suitable selected data structure. An adaptive selection of data structures is a separate topic that has to be implemented in a distinct library.

* automatically **optimizing concurrent data accesses**.

* **handling** or hiding differences in arithmetic operations. For example, due to **different rounding** or different implementations of floating point operations, results can differ slightly between accelerators.

* **guaranteeing any determinism** of results. Due to the freedom of the library to reorder or repartition the threads within the tasks it is not possible or even desired to preserve deterministic results. For example, the non-associativity of floating point operations give non-deterministic results within and across accelerators.

The *alpaka* library is aimed at parallelization within nodes of a cluster.
It does not compete with libraries for distribution of processes across nodes and communication among those.
For these purposes libraries like MPI (Message Passing Interface) or others should be used.
MPI is situated one layer higher and can be combined with *alpaka* to facilitate the hardware of a whole heterogeneous cluster.
The *alpaka* library can be used for parallelization within nodes, MPI for parallelization across nodes.
