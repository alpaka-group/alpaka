# Rationale

## Kernels

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

### Implementation Variants

There are two possible ways to tell the kernel about the accelerator type:
 1. The kernel is templated on the accelerator type
  * + This allows users to specialize them for different accelerators. (Is this is really necessary or desired?)
  * - The kernel has to be a class template. This does not allow C++ lambdas to be used as kernels because they are no templates themselves (but only their `operator()` can be templated in C++14).
  * - This prevents the user from instantiating an accelerator independent kernel before executing it and then adapting it to the given accelerator on execution.
  Because the memory layout in inheritance hierarchies is undefined a simple copy of the user kernel or its members to its specialized type is not possible platform independently.
  This would require a copy from UserKernel<TDummyAcc> to UserKernel<TAcc> to be possible.
  The only way to allow this would be to require the user to implement a templated copy constructor for every kernel.
  This is not allowed for kernels that should be copyable to a CUDA device because std::is_trivially_copyable requires the kernel to have no non-trivial copy constructors.
  * a) and inherits from the accelerator. 
    * +/- To give a device function called from the kernel functor access to the accelerator methods, these methods have to be templated on the accelerated kernel and get a reference to the accelerator.
    This allows to give them access not only to the accelerator methods but also to the other kernel methods.
    This is inconsistent because the kernel uses inheritance and subsequent function calls get a parameter.
    * - The kernel itself has to inherit at least protected from the accelerator to allow the KernelExecutor to access the Accelerator.
    * - How do accelerator functions called from the kernel (and not within the kernel class itself) access the accelerator methods?
    Casting this to the accelerator type and giving it as parameter is too much to require from the user.
  * b) and has a reference to the accelerator as parameter.
    * + This allows to use the accelerator in accelerator functions called from the kernel (and not within the kernel class itself) to access the accelerator methods in the same way the kernel entry point function can.
    * - This would require an additional object in device memory taking up valuable CUDA registers.
    * TODO: Will all the device functions be inlined nevertheless (because we do not use run-time polymorphism)? This would make it a non-reason.
 2. The `operator()` is templated on the accelerator type and has a reference to the accelerator as parameter.
  * + The kernel can be an arbitrary function object with ALPAKA_FCT_HOST_ACC attributes.
  * + This would allow to instantiate the accelerator independent kernel and set its members before execution.
  * +/- C++14 provides polymorphic lambdas. All compilers (even MSVC) support this. Inheriting from a non capturing lambda for the KernelExecutor is allowed. (TODO: How to check for a non capturing lambda?)
  * - The `operator()` could be overloaded on the accelerator type but not the kernel itself, so it always has the same members.
  * - This would require an additional object in device memory taking up valuable CUDA registers.
    TODO: Will all the device functions be inlined nevertheless (because we do not use run-time polymorphism)? This would make it a non-reason.

### Implementation Notes

Currently we implement version 1b).

Kernels bound to an accelerator can be built with the `createKernelExecutor` template function.
This function returns an object that stores the given kernel type and the constructor argumnts..
To separate the kernel execution attributes (grid/block-extents, stream) from the invocation arguments, the first call to `operator()` returns a kernel executor with stored execution attributes.
The returned executor can then be executed with the `operator()` leading to `createKernelExecutor<TAcc, TKernel>(TKernelConstrArgs ... args)(<grid/block>-extents, stream = 0)(invocation-args ...)` for a complete kernel invocation.

TODO: Why do we require the user to have a default template argument `template<typename TAcc = alpaka::IAcc<>>` for the kernel? 
 - Because we can not create a kernel before binding it to an accelerator we could just make it a simple template `template<typename TAcc>` and make IAcc an implementation detail. 
 - If it is because the possible use of boost::mpl::_1, would it be better to not require the usage of IAcc and make it an implementation detail requiring the user to use `template<typename TAcc = boost::mpl::_1>` directly?
 - Why is boost::mpl::_1 even required in this case?
 
### External Block Shared Memory

The size of the external block shared memory has to be available at compile time.

TODO: Why?

TODO: Explain trait.

## Accelerators

All the accelerators are restricted by the possibilities of CUDA.

The library does not use a common accelerator base class with virtual functions from which all accelerator implementations inherit (run-time polymorphism).
This is required because in the case of CUDA copying objects (kernels inheriting from the accelerator) with virtual functions into device memory is not viable because of possibly incompatible object layout (std::is_trivially_copyable).
To deliver a common accelerator interface static polymorphism is used instead.

The accelerator interface IAcc hides all implementation details of the underlying accelerator base class by protected inheritance.
Private inheritance is not possible, because the `KernelExecutor` implementation for a special accelerator sometimes needs access to the accelerator itself in `KernelExecutor<IAcc<Acc>>`.

TODO: Add note about ALPAKA_FCT_HOST_ACC!

## Accelerator Implementation Notes

### Serial

The serial accelerator is only for debugging purposes because it only allows blocks with exactly one kernel.
Therefore it does not implement real synchronization or atomic primitives.

### Threads

#### Execution

To prevent recreation of the threads between execution of different blocks in the grid, the threads are stored inside a thread pool.
This thread pool is local to the invocation because making it local to the KernelExecutor could mean a heavy memory usage and lots of idling threads when there are multiple KernelExecutors around.
Because the default policy of the threads in the pool is to yield instead of waiting, this would also slow down the system immensely.

### Fibers

#### Execution

To prevent recreation of the fibers between execution of different blocks in the grid, the fibers are stored inside a fibers pool.
This fibers pool is local to the invocation because making it local to the KernelExecutor could mean a heavy memory usage when there are multiple KernelExecutors around.

### OpenMP

#### Execution

Parallel execution of the kernels in a block is required because when syncBlockThreads is called all of them have to be done with their work up to this line.
So we have to spawn one real thread per kernel in a block.
`omp for` is not useful because it is meant for cases where multiple iterations are executed by one thread but in our case a 1:1 mapping is required.
Therefore we use `omp parallel` with the specified number of threads in a block.
Another reason for not using `omp for` like `#pragma omp parallel for collapse(3) num_threads(blockDim.x*blockDim.y*blockDim.z)` is that `#pragma omp barrier` used for intra block synchronization is not allowed inside `omp for` blocks.

Because OpenMP is designed for a 1:1 abstraction of hardware to software threads, the block size is restricted by the number of OpenMP threads allowed by the runtime. 
This could be as little as 2 or 4 kernels but on a system with 4 cores with hyperthreading OpenMP can for example also allow a maximum of 64 threads.

#### Index

OpenMP only provides a linear thread index. This index is converted to a 3 dimensional index at runtime.

#### Atomic

We can not use '#pragma omp atomic' because braces or calling other functions directly after `#pragma omp atomic` are not allowed!
Because we are implementing the CUDA atomic operations which return the old value, this requires `#pragma omp critical` to be used.
`omp_set_lock` is an alternative but is usually slower.

### CUDA
