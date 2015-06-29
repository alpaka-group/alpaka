Rationale
=========

This document gives reasons why some implementation details are the way they are.


Kernel Interface
----------------

### Requirements

- User kernels should be implemented independent of the accelerator.
- A user kernel has to have access to accelerator methods like synchronization within blocks, index retrieval and many more.
- For usage with CUDA the kernel methods have to be attributed with \__device\__ \__host\__.
- The user kernel has to fulfill std::is_trivially_copyable because only such objects can be copied into CUDA device memory.
  A trivially copyable class is a class that
   1. Has no non-trivial copy constructors(this also requires no virtual functions or virtual bases)
   2. Has no non-trivial move constructors
   3. Has no non-trivial copy assignment operators
   4. Has no non-trivial move assignment operators
   5. Has a trivial destructor

### Implementation variants

There are two possible ways to tell the kernel about the accelerator type:
 1. The kernel is templated on the accelerator type
  * + This allows users to specialize them for different accelerators. (Is this is really necessary or desired?)
  * - The kernel has to be a class template. This does not allow C++ lambdas to be used as kernels because they are no templates themselves (but only their `operator()` can be templated in C++14).
  * - This prevents the user from instantiating an accelerator independent kernel before executing it.
  Because the memory layout in inheritance hierarchies is undefined a simple copy of the user kernel or its members to its specialized type is not possible platform independently.
  This would require a copy from UserKernel<TDummyAcc> to UserKernel<TAcc> to be possible.
  The only way to allow this would be to require the user to implement a templated copy constructor for every kernel.
  This is not allowed for kernels that should be copyable to a CUDA device because std::is_trivially_copyable requires the kernel to have no non-trivial copy constructors.
  * a) and inherits from the accelerator. 
    * +/- To give a device function called from the kernel function object access to the accelerator methods, these methods have to be templated on the kernel function object and get a reference to the accelerator.
    This allows to give them access not only to the accelerator methods but also to the other kernel methods.
    This is inconsistent because the kernel uses inheritance and subsequent function calls get a parameter.
    * - The kernel itself has to inherit at least protected from the accelerator to allow the KernelExecutor to access the Accelerator.
    * - How do accelerator functions called from the kernel (and not within the kernel class itself) access the accelerator methods?
    Casting this to the accelerator type and giving it as parameter is too much to require from the user.
  * b) and the `operator()` has a reference to the accelerator as parameter.
    * + This allows to use the accelerator in accelerator functions called from the kernel (and not within the kernel class itself) to access the accelerator methods in the same way the kernel entry point function can.
    * - This would require an additional object (the accelerator) in device memory taking up valuable CUDA registers (opposed to the inheritance solution). At least on CUDA all the accelerator functions could be inlined nevertheless.
 2. The `operator()` is templated on the accelerator type and has a reference to the accelerator as parameter.
  * + The kernel can be an arbitrary function object with ALPAKA_FCT_HOST_ACC attributes.
  * + This would allow to instantiate the accelerator independent kernel and set its members before execution.
  * +/- C++14 provides polymorphic lambdas. All compilers (even MSVC) support this. Inheriting from a non capturing lambda for the KernelExecutor is allowed. (TODO: How to check for a non capturing lambda?)
  * - The `operator()` could be overloaded on the accelerator type but not the kernel itself, so it always has the same members.
  * - This would require an additional object (the accelerator) in device memory taking up valuable CUDA registers (opposed to the inheritance solution). At least on CUDA all the accelerator functions could be inlined nevertheless.

### Implementation notes

Currently we implement version 2.

A kernel executor can be obtained by calling `alpaka::exec::create<TAcc>(TWorkDiv, TStream)` with the execution attributes (grid/block-extents, stream).
This separates the kernel execution attributes (grid/block-extents, stream) from the invocation arguments.
The returned executor can then be called with the `operator()` leading to `alpaka::exec::create<TAcc>(TWorkDiv, TStream)(invocation-args ...)` for a complete kernel invocation.
 

Block Shared Memory
-------------------
 
### Internal Block Shared Memory

The size of block shared memory that is allocated inside the kernel is required to be given as compile time constant.
This is due to CUDA not allowing to allocate block shared memory inside a kernel at runtime.
 
### External Block Shared Memory

The size of the external block shared memory is obtained from a trait that can be specialized for each kernel.
The trait is called with the current kernel invocation parameters and the block size prior to each kernel execution.
Because the block shared memory size is only ever constant or dependent on the block size or the parameters of the invocation this has multiple advantages:
* It forces the separation of the kernel invocation from the calculation of the required block shared memory size.
* It lets the user write this calculation once instead of multiple time spread across the code.


Accelerators
------------

All the accelerators are restricted by the possibilities of CUDA.

The library does not use a common accelerator base class with virtual functions from which all accelerator implementations inherit (run time polymorphism).
This reduces runtime overhead because everything can be checked at compile time.

**TODO**: Add note about ALPAKA_FCT_HOST_ACC!


Accelerator Access within Kernels
---------------------------------

CUDA always tracks some implicit state like the current device in host code or the current thread and block index in kernel code.
This implicit state hides dependencies and can produce bugs if the wrong device is active during a memory operation in host code or similar things.
In alpaka this is always made explicit.
Streams, events and memory always require a device parameter for their creation.

The kernels have access to the accelerator through a reference parameter.
There are two possible ways to implement access to accelerator dependent functionality inside a kernel:
* Making the functions/templates members of the accelerator (maybe by inheritance) and calling them like `acc.syncThreads()` or `acc.template getIdx<Grid, Thread, Dim1>()`.
This would require the user to know and understand when to use the template keyword inside dependent type  object function calls.
* The functions are only light wrappers around traits that can be specialized taking the accelerator as first value (it can not be the last value because of the potential use of variadic arguments). 
The resulting code would look like `sync(acc)` or `getIdx<Grid, Thread, Dim1>(acc)`.
Internally these wrappers would call trait templates that are specialized for the specific accelerator e.g. `template<typename TAcc> Sync{...};`

The second version is easier to understand and usually shorter to use in user code.
NOTE: Currently version 1 is implemented!


Accelerator Implementation Notes
--------------------------------

### Serial

The serial accelerator only allows blocks with exactly one thread.
Therefore it does not implement real synchronization or atomic primitives.

### Threads

#### Execution

To prevent recreation of the threads between execution of different blocks in the grid, the threads are stored inside a thread pool.
This thread pool is local to the invocation because making it local to the KernelExecutor could mean a heavy memory usage and lots of idling kernel-threads when there are multiple KernelExecutors around.
Because the default policy of the threads in the pool is to yield instead of waiting, this would also slow down the system immensely.

std::thread::hardware_concurrency()

### Fibers

#### Execution

To prevent recreation of the fibers between execution of different blocks in the grid, the fibers are stored inside a fibers pool.
This fiber pool is local to the invocation because making it local to the KernelExecutor could mean a heavy memory usage when there are multiple KernelExecutors around.

### OpenMP

#### Execution

Parallel execution of the kernels in a block is required because when syncBlockThreads is called all of them have to be done with their work up to this line.
So we have to spawn one real thread per kernel in a block.
`omp for` is not useful because it is meant for cases where multiple iterations are executed by one thread but in our case a 1:1 mapping is required.
Therefore we use `omp parallel` with the specified number of threads in a block.
Another reason for not using `omp for` like `#pragma omp parallel for collapse(3) num_threads(blockDim.x*blockDim.y*blockDim.z)` is that `#pragma omp barrier` used for intra block synchronization is not allowed inside `omp for` blocks.

Because OpenMP is designed for a 1:1 abstraction of hardware to software threads, the block size is restricted by the number of OpenMP threads allowed by the runtime. 
This could be as little as 2 or 4 kernels but on a system with 4 cores and hyper-threading OpenMP can also allow 64 threads.

::omp_get_max_threads()
::omp_get_thread_limit()

#### Index

OpenMP only provides a linear thread index. This index is converted to a 3 dimensional index at runtime.

#### Atomic

We can not use '#pragma omp atomic' because braces or calling other functions directly after `#pragma omp atomic` are not allowed.
Because we are implementing the CUDA atomic operations which return the old value, this requires `#pragma omp critical` to be used.
`omp_set_lock` is an alternative but is usually slower.

### CUDA

Nearly all CUDA functionality can be directly mapped to alpaka function calls.
A major difference is that CUDA requires the block and grid sizes to be given in (x, y, z) order.
Alpaka uses the mathematical C/C++ array indexing scheme [z][y][x].
Dimension 0 in this case is z, dimensions 2 is x.

Furthermore alpaka does not require the indices and extents to be 3-dimensional.
The accelerators are templatized on and support arbitrary dimensionality.
NOTE: Currently the CUDA implementation is restricted to a maximum of 3 dimensions!

NOTE: The CUDA-accelerator back-end can change the current CUDA device and will NOT set the device back to the one prior to the invocation of the alpaka function!

Device Implementations
----------------------

|-|CPU|CUDA|
|---|---|---|
|Devices|(only ever one device)|cudaGetDeviceCount, cudaGetDevice, cudaGetDeviceProperties|
|Events|std::condition_variable, std::mutex|cudaEventCreateWithFlags, cudaEventDestroy, cudaEventRecord, cudaEventSynchronize, cudaEventQuery|
|Streams|Thread-Pool with exactly one worker|cudaStreamCreateWithFlags, cudaStreamDestroy, cudaStreamQuery, cudaStreamSynchronize, cudaStreamWaitEvent|
|Memory|new , delete[], std::memcpy, std::memset, (cudaHostRegister, cudaHostUnregister for memory pinning if available)|cudaMalloc, cudaFree, cudaMemcpy, cudaMemset, cudaHostRegister, cudaHostUnregister, cudaHostGetDevicePointer|
|RNG|std::mt19937, std::normal_distribution, std::uniform_real_distribution, std::uniform_int_distribution|curand_init, curandStateXORWOW_t, curand, curand_normal, curand_normal_double, curand_uniform, curand_uniform_double|
