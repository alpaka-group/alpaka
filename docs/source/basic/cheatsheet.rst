Cheatsheet
==========

.. raw:: pdf

   Spacer 0,1

.. only:: html

   Download pdf version `here <../cheatsheet.pdf>`_

General
-------

- Getting alpaka: https://github.com/alpaka-group/alpaka
- Issue tracker, questions, support: https://github.com/alpaka-group/alpaka/issues
- All alpaka names are in namespace alpaka and header file `alpaka/alpaka.hpp`
- This document assumes

  .. code-block:: c++

     #include <alpaka/alpaka.hpp>
     namespace myProject {
         using namespace alpaka;
     }

.. warning::

  Using ``using namespace alpaka;`` is global namespace should be avoided, due to possible side effects with other libraries.

All methods and classes in the `alpaka` namespace can be called from the cpu controller thread (named `host`) and from the compute device.

- `alpaka::onHost` can only be called from `host`.
- `alpaka::onAcc` can only be called from within a kernel running on the compute device.

Methods starting with `onHost::make` (e.g., `onHost::makeHostDevice()`) create handles to instances where the copy is only a shallow copy and not a deep copy.
Methods starting with `get` (e.g., `onHost::getExtents(...)`) provide access to properties of an instance.

Accelerator, Platform and Device
--------------------------------

Define in-kernel thread indexing type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  .. literalinclude:: ../../snippets/cheatsheet/cheatsheet.cpp
    :language: cpp
    :start-after: BEGIN-CHEATSHEET-init
    :end-before: END-CHEATSHEET-init
    :dedent:

Usage of multi-dimensional vectors required for extents or indexing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  .. literalinclude:: ../../snippets/cheatsheet/cheatsheet.cpp
    :language: cpp
    :start-after: BEGIN-CHEATSHEET-vectorCreate
    :end-before: END-CHEATSHEET-vectorCreate
    :dedent:

Access components of and destructure multi-dimensional indices and extents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  .. literalinclude:: ../../snippets/cheatsheet/cheatsheet.cpp
    :language: cpp
    :start-after: BEGIN-CHEATSHEET-vectorAccess
    :end-before: END-CHEATSHEET-vectorAccess
    :dedent:

Linearize multi-dimensional vectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  .. literalinclude:: ../../snippets/cheatsheet/cheatsheet.cpp
    :language: cpp
    :start-after: BEGIN-CHEATSHEET-linearize
    :end-before: END-CHEATSHEET-linearize
    :dedent:

Map linear index to multi-dimensional index
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  .. literalinclude:: ../../snippets/cheatsheet/cheatsheet.cpp
    :language: cpp
    :start-after: BEGIN-CHEATSHEET-mapToMd
    :end-before: END-CHEATSHEET-mapToMd
    :dedent:

Available apis
~~~~~~~~~~~~~~
  .. code-block:: c++

    api::host
    api::cuda
    api::hip
    api::oneApi

Device kinds
~~~~~~~~~~~~
  .. code-block:: c++

     deviceKind::cpu
     deviceKind::amdGpu
     deviceKind::nvidiaGpu
     deviceKind::intelGpu

Executors
~~~~~~~~~
  .. code-block:: c++

     exec::cpuSerial
     exec::cpuOmpBlocks
     exec::cpuTbbBlocks
     exec::gpuCuda
     exec::gpuHip
     exec::oneApi

Create device selector and select a device by index
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  .. literalinclude:: ../../snippets/cheatsheet/cheatsheet.cpp
    :language: cpp
    :start-after: BEGIN-CHEATSHEET-makeDevice
    :end-before: END-CHEATSHEET-makeDevice
    :dedent:

Queue and Events
----------------

Create a queue for a device
~~~~~~~~~~~~~~~~~~~~~~~~~~~
  .. literalinclude:: ../../snippets/cheatsheet/cheatsheet.cpp
    :language: cpp
    :start-after: BEGIN-CHEATSHEET-makeQueue
    :end-before: END-CHEATSHEET-makeQueue
    :dedent:

Put a task for execution
~~~~~~~~~~~~~~~~~~~~~~~~
  .. literalinclude:: ../../snippets/cheatsheet/cheatsheet.cpp
    :language: cpp
    :start-after: BEGIN-CHEATSHEET-enqueueHostTask
    :end-before: END-CHEATSHEET-enqueueHostTask
    :dedent:

Wait for all operations in the queue
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  .. literalinclude:: ../../snippets/cheatsheet/cheatsheet.cpp
    :language: cpp
    :start-after: BEGIN-CHEATSHEET-waitQueue
    :end-before: END-CHEATSHEET-waitQueue
    :dedent:

Check if a queue is empty
~~~~~~~~~~~~~~~~~~~~~~~~~
  .. literalinclude:: ../../snippets/cheatsheet/cheatsheet.cpp
    :language: cpp
    :start-after: BEGIN-CHEATSHEET-isQueueEmpty
    :end-before: END-CHEATSHEET-isQueueEmpty
    :dedent:

Create an event
~~~~~~~~~~~~~~~
  .. literalinclude:: ../../snippets/cheatsheet/cheatsheet.cpp
    :language: cpp
    :start-after: BEGIN-CHEATSHEET-makeEvent
    :end-before: END-CHEATSHEET-makeEvent
    :dedent:

Put an event to the queue
~~~~~~~~~~~~~~~~~~~~~~~~~
  .. literalinclude:: ../../snippets/cheatsheet/cheatsheet.cpp
    :language: cpp
    :start-after: BEGIN-CHEATSHEET-enqueueEvent
    :end-before: END-CHEATSHEET-enqueueEvent
    :dedent:

Check if the event is completed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  .. literalinclude:: ../../snippets/cheatsheet/cheatsheet.cpp
    :language: cpp
    :start-after: BEGIN-CHEATSHEET-eventIsComplete
    :end-before: END-CHEATSHEET-eventIsComplete
    :dedent:

Wait for the event (and all operations put to the same queue before it)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  .. literalinclude:: ../../snippets/cheatsheet/cheatsheet.cpp
    :language: cpp
    :start-after: BEGIN-CHEATSHEET-waitEvent
    :end-before: END-CHEATSHEET-waitEvent
    :dedent:

Memory
------

Memory allocation and transfers are symmetric for host and devices, both done via alpaka API

Allocate a shared buffer in host memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  .. literalinclude:: ../../snippets/cheatsheet/cheatsheet.cpp
    :language: cpp
    :start-after: BEGIN-CHEATSHEET-allocHostBuffer
    :end-before: END-CHEATSHEET-allocHostBuffer
    :dedent:

Create a view to host memory represented by a pointer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  .. literalinclude:: ../../snippets/cheatsheet/cheatsheet.cpp
    :language: cpp
    :start-after: BEGIN-CHEATSHEET-makeViewFromPtr
    :end-before: END-CHEATSHEET-makeViewFromPtr
    :dedent:

Create a view to host std::vector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  .. literalinclude:: ../../snippets/cheatsheet/cheatsheet.cpp
    :language: cpp
    :start-after: BEGIN-CHEATSHEET-makeViewFromStdVector
    :end-before: END-CHEATSHEET-makeViewFromStdVector
    :dedent:

Create a view to host std::array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  .. literalinclude:: ../../snippets/cheatsheet/cheatsheet.cpp
    :language: cpp
    :start-after: BEGIN-CHEATSHEET-makeViewStdArray
    :end-before: END-CHEATSHEET-makeViewStdArray
    :dedent:

Get a raw pointer to a view initialization, etc.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  .. literalinclude:: ../../snippets/cheatsheet/cheatsheet.cpp
    :language: cpp
    :start-after: BEGIN-CHEATSHEET-dataPtr
    :end-before: END-CHEATSHEET-dataPtr
    :dedent:

Get the pitches of a view
~~~~~~~~~~~~~~~~~~~~~~~~~
  .. literalinclude:: ../../snippets/cheatsheet/cheatsheet.cpp
    :language: cpp
    :start-after: BEGIN-CHEATSHEET-getPitches
    :end-before: END-CHEATSHEET-getPitches
    :dedent:

View initialization, etc.
~~~~~~~~~~~~~~~~~~~~~~~~~
  .. literalinclude:: ../../snippets/cheatsheet/cheatsheet.cpp
    :language: cpp
    :start-after: BEGIN-CHEATSHEET-initView
    :end-before: END-CHEATSHEET-initView
    :dedent:

Allocate a buffer
~~~~~~~~~~~~~~~~~
  .. literalinclude:: ../../snippets/cheatsheet/cheatsheet.cpp
    :language: cpp
    :start-after: BEGIN-CHEATSHEET-allocBuffer
    :end-before: END-CHEATSHEET-allocBuffer
    :dedent:

Copy multidimensional buffer/view or span data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  .. literalinclude:: ../../snippets/cheatsheet/cheatsheet.cpp
    :language: cpp
    :start-after: BEGIN-CHEATSHEET-memcpy
    :end-before: END-CHEATSHEET-memcpy
    :dedent:

Allocate a buffer with the same extents from a std::vector or std::array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  .. literalinclude:: ../../snippets/cheatsheet/cheatsheet.cpp
    :language: cpp
    :start-after: BEGIN-CHEATSHEET-allocLike
    :end-before: END-CHEATSHEET-allocLike
    :dedent:

Kernel Execution
----------------

Manually set a kernel launch configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  .. literalinclude:: ../../snippets/cheatsheet/cheatsheet.cpp
    :language: cpp
    :start-after: BEGIN-CHEATSHEET-manualFrameSpec
    :end-before: END-CHEATSHEET-manualFrameSpec
    :dedent:

Automatically select a valid kernel launch configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  .. literalinclude:: ../../snippets/cheatsheet/cheatsheet.cpp
    :language: cpp
    :start-after: BEGIN-CHEATSHEET-autoFrameSpec
    :end-before: END-CHEATSHEET-autoFrameSpec
    :dedent:

Kernel Implementation
---------------------

Define a kernel as a C++ functor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ``ALPAKA_FN_ACC`` is required for kernels and functions called inside, ``acc`` is mandatory first parameter, its type is the template parameter.
  ``acc`` must be a constant reference.

  .. literalinclude:: ../../snippets/cheatsheet/cheatsheet.cpp
    :language: cpp
    :start-after: BEGIN-CHEATSHEET-myKernel
    :end-before: END-CHEATSHEET-myKernel
    :dedent:

Instantiate a kernel (does not launch it yet)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  acc parameter of the kernel is provided automatically, does not need to be specified here

  .. literalinclude:: ../../snippets/cheatsheet/cheatsheet.cpp
    :language: cpp
    :start-after: BEGIN-CHEATSHEET-createKernelWithArg
    :end-before: END-CHEATSHEET-createKernelWithArg
    :dedent:

Put the kernel for execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  .. literalinclude:: ../../snippets/cheatsheet/cheatsheet.cpp
    :language: cpp
    :start-after: BEGIN-CHEATSHEET-enqueueKernel
    :end-before: END-CHEATSHEET-enqueueKernel
    :dedent:

Access multi-dimensional indices and extents of blocks, threads, and elements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  .. code-block:: c++

     // origin: grid, block
     // unit: blocks, threads
     auto idxMd = acc.getIdxWithin(onAcc::origin::*, onAcc::unit::*);
     auto extentMd = acc.getExtentsOf(onAcc::origin::*, alpaka::onAcc::unit::*);

Allocate static shared memory variable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  .. literalinclude:: ../../snippets/cheatsheet/cheatsheet.cpp
    :language: cpp
    :start-after: BEGIN-CHEATSHEET-staticSharedMem
    :end-before: END-CHEATSHEET-staticSharedMem
    :dedent:

Get dynamic shared memory pool, requires the kernel to have a data member with the size in bytes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  .. literalinclude:: ../../snippets/cheatsheet/cheatsheet.cpp
    :language: cpp
    :start-after: BEGIN-CHEATSHEET-dynSharedMem
    :end-before: END-CHEATSHEET-dynSharedMem
    :dedent:

Or must specialize a trait for the kernel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  .. literalinclude:: ../../snippets/cheatsheet/cheatsheet.cpp
    :language: cpp
    :start-after: BEGIN-CHEATSHEET-dynSharedMemTrait
    :end-before: END-CHEATSHEET-dynSharedMemTrait
    :dedent:

Synchronize threads of the same block
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  .. literalinclude:: ../../snippets/cheatsheet/cheatsheet.cpp
    :language: cpp
    :start-after: BEGIN-CHEATSHEET-inKernelBlockWait
    :end-before: END-CHEATSHEET-inKernelBlockWait
    :dedent:

Atomic operations
~~~~~~~~~~~~~~~~~
  .. literalinclude:: ../../snippets/cheatsheet/cheatsheet.cpp
    :language: cpp
    :start-after: BEGIN-CHEATSHEET-atomicAdd
    :end-before: END-CHEATSHEET-atomicAdd
    :dedent:

Memory fences on block-, device- or system level (guarantees LoadLoad and StoreStore ordering)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  .. literalinclude:: ../../snippets/cheatsheet/cheatsheet.cpp
    :language: cpp
    :start-after: BEGIN-CHEATSHEET-memFence
    :end-before: END-CHEATSHEET-memFence
    :dedent:

Math functions
~~~~~~~~~~~~~~
  .. literalinclude:: ../../snippets/cheatsheet/cheatsheet.cpp
    :language: cpp
    :start-after: BEGIN-CHEATSHEET-math
    :end-before: END-CHEATSHEET-math
    :dedent:

Similar for other math functions.
