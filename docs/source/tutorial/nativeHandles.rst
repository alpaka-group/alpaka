Using Native Backend Handles
============================

*alpaka* devices and queues are created by *alpaka* through :ref:`device selection <device-selection>` and the ``makeQueue`` method.
Applications that already manage a vendor context and its streams, such as PyTorch, or that exchange work with a third-party library, need to reuse the existing native objects instead.
Linking already existing devices or queues also simplify porting existing CUDA/HIP applications to alpaka.
Creating a second context or stream for the same device breaks the ordering the application established and adds a context switch that is easy to avoid.
The link API adopts an existing native device, stream or queue, so that *alpaka* objects can operate on them.

There is **no** native handle for the host API, and the ``link*`` methods are therefore not available for it.

Linking a device
----------------

The device selector is created for a concrete API and device kind, as in the :ref:`device selection <device-selection>` section.
Pass the native device to ``linkDevice`` instead of calling ``makeDevice``.
For CUDA and HIP the native device is the device ordinal, for SYCL it is a pair of ``sycl::device`` and ``sycl::context``.

Linking a queue
---------------

The native stream or queue must be asynchronous.
*alpaka* creates its own streams with the non-blocking flag and assumes asynchronous semantics.
The *alpaka* queue kind is independent of that: ``queueKind::blocking`` is supported and makes *alpaka* synchronize after each enqueue.
Linking the CUDA/HIP legacy default stream is rejected.

The caller owns the native stream or queue.
*alpaka* never destroys or resets an adopted device, stream, or queue.
The native handle must outlive every *alpaka* handle that refers to it.
When you no longer need the *alpaka* registration, call ``unlinkQueue`` to remove it.
This does not destroy the *alpaka* queue handles you may hold, and it does not touch the native stream/queue.

The ``getNativeHandle`` accessors on the *alpaka* device and queue still return the native objects, so you can pass them to vendor libraries at any point.

Lifetime and synchronization
----------------------------

The ``syncBeforeDestroy`` argument of ``linkDevice`` and ``linkQueue`` defaults to ``true``.
It only controls a synchronization that *alpaka* performs before it drops its last reference to the wrapper.
Pass ``false`` when the caller synchronizes the native object and destroys it itself, so the call is safe.
A linked device with ``syncBeforeDestroy=true`` waits on all its *alpaka* queues, including linked queues created from that device.

Destroy the *alpaka* handles before the native handles.
``unlinkQueue`` and ``unlinkDevice`` only remove *alpaka*'s registration; the caller's *alpaka* handle copies keep the wrapper alive until they go out of scope.
If you drop the native stream while an *alpaka* queue still refers to it, the *alpaka* object reads a dangling handle.

Backend notes
-------------

- CUDA and HIP identify a device by its ordinal, SYCL by a ``(device, context)`` pair.
- A SYCL queue linked as an *alpaka* queue must be in-order.
  ``linkQueue`` throws for an out-of-order queue.
- A SYCL device linked through ``linkDevice`` cannot be mixed with an *alpaka*-managed device on the same platform.
  Linking throws if a managed device for that platform already exists.
- For CUDA and HIP, ``linkDevice`` throws if the device ordinal is already an *alpaka*-managed device, because a managed device cannot be unlinked.

Full examples
-------------

The examples below are the compiled tests.
They create the native handles, link the device and queue, run work, and destroy the native handles after the alpaka wrappers.

CUDA
~~~~

.. literalinclude:: ../../snippets/example/210_nativeHandles.cpp
   :language: cpp
   :start-after: BEGIN-TUTORIAL-nativeCuda
   :end-before: END-TUTORIAL-nativeCuda
   :dedent:

HIP
~~~

.. literalinclude:: ../../snippets/example/210_nativeHandles.cpp
   :language: cpp
   :start-after: BEGIN-TUTORIAL-nativeHip
   :end-before: END-TUTORIAL-nativeHip
   :dedent:

SYCL/oneAPI
~~~~~~~~~~~

.. literalinclude:: ../../snippets/example/210_nativeHandles.cpp
   :language: cpp
   :start-after: BEGIN-TUTORIAL-nativeOneApi
   :end-before: END-TUTORIAL-nativeOneApi
   :dedent:

Complete source file
--------------------

.. raw:: html

   <details class="full-source">
   <summary>210_nativeHandles.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/210_nativeHandles.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
   <br/>
