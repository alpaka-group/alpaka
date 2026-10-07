.. only:: html

  .. image:: ../logo/alpaka.svg

.. only:: latex

  .. image:: ../logo/alpaka.png

*alpaka - An Abstraction Library for Parallel Kernel Acceleration*

Its aim is to provide performance portability across accelerators by abstracting the underlying levels of parallelism.

The library is platform-independent and supports the concurrent and cooperative use of multiple devices, including host CPUs (x86, ARM, RISC-V, and Power8+) and GPUs from different vendors (NVIDIA, AMD, and Intel).
A variety of accelerator backends—CUDA, HIP, SYCL, OpenMP, and serial execution—are available and can be selected based on the target device.
Only a single implementation of a user kernel is required, expressed as a function object with a standardized interface.
This eliminates the need to write specialized CUDA, HIP, SYCL, OpenMP, or threading code.
Moreover, multiple accelerator backends can be combined to target different vendor hardware within a single system and even within a single application.

.. CAUTION::
   The readthedocs pages are provided with best effort, but may contain outdated sections.

alpaka - How to Read This Document
----------------------------------

Generally, **follow the manual pages in-order** to get started.
Individual chapters are based on the information of the chapters before.

.. only:: html

   The online version of this document is **versioned** and shows by default the manual of the last *stable* version of alpaka.
   If you are looking for the latest *development* version, `click here <https://alpaka.readthedocs.io/en/latest/>`_.

.. note::

   Are you looking for our latest user Doxygen docs for the API?

   - See https://alpaka.readthedocs.io/en/latest/doxygen/namespaces.html

.. toctree::
   :caption: Basic
   :maxdepth: 2

   basic/intro.rst
   basic/install.rst
   basic/example.rst
   basic/terms.rst
   basic/abstraction.rst
   basic/library.rst
   basic/cheatsheet.rst

.. toctree::
   :caption: Tutorial - Getting Started
   :maxdepth: 1
   :titlesonly:

   tutorial/motivation.rst
   tutorial/vector.rst
   tutorial/device.rst
   tutorial/queue.rst
   tutorial/memoryAllocation.rst
   tutorial/memoryOperations.rst
   tutorial/kernel.rst

.. toctree::
   :caption: Tutorial - Advanced
   :maxdepth: 1
   :titlesonly:

   tutorial/views.rst
   tutorial/multidim.rst
   tutorial/kernelParallelism.rst
   tutorial/synchronization.rst
   tutorial/sharedMemory.rst
   tutorial/math.rst
   tutorial/atomics.rst
   tutorial/memFence.rst
   tutorial/events.rst
   tutorial/random.rst
   tutorial/intrinsics.rst
   tutorial/warp.rst
   tutorial/kernelFn.rst
   tutorial/vendorInterop.rst
   tutorial/algorithms.rst

.. toctree::
   :caption: Advanced
   :maxdepth: 2

   advanced/cmake.rst
   advanced/datastorage.rst
   advanced/benchmark.rst
   advanced/crosscompile.rst

.. toctree::
   :caption: Developer
   :maxdepth: 2

   dev/logging.rst
   dev/online_tools.rst

.. toctree::
   :caption: Contribution
   :maxdepth: 2

   contribution/sphinx.rst
   contribution/tools.rst

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :titlesonly:

   doxygen
   doxygen_dev

Indices and Tables
==================

* :ref:`genindex`
* :ref:`search`
