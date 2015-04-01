Abstraction
===========

Parallelism and memory hierarchies at all levels need to be exploited in order to achieve performance portability across various types of accelerators.

On the hardware side we have nodes with multiple sockets/processors extended by accelerators like GPUs or Intel Xeon Phi each with their own processing units.
Within a CPU or a Intel Xeon Phi there are cores with hyper-threads and vector units, within a GPU there are many small cores.
Each of those entities has access to different memories in the hierarchy.
For example each socket/processor manages its RAM and the cores additionally have access to L3, L2 and L1 caches.
On a GPU there is global, constant and shared memory.
**alpaka** is designed to abstract from these differences without sacrificing speed by defining a domain decomposition for the computation domain.
This domain decomposition is abstract enough to map optimally to all (currently known) accelerators.
The **alpaka** library hides the mapping to the underlying hardware together with the execution primitives like kernel-threads and fibers allowing accelerator independent performance and portability.

A process running on a multi-socket node is the largest entity within **alpaka**.
The library itself only abstracts the task and data parallel execution on the process/node level and down.
It does not provide any primitives for inter-node communication but such libraries can build upon **alpaka**.
The process always has a main thread and is by definition running on the host.
It can access the host memory and various accelerator devices.
Such accelerator devices can be GPUs, Intel Xeon Phi, the host itself or other hardware.
Thus the host not necessarily has to be different from the accelerator device used to compute on.
For example a Intel Xeon Phi simultaneously can be the host and the accelerator device.
**alpaka** also allows the parallel execution on nodes without any accelerator hardware.
The node the process is running on can itself be used as an accelerator device.

**alpaka** can be used to offload the parallel execution of task and data parallel work simultaneously onto different accelerator devices.


Task Parallelism
----------------

With the concept of streams known from CUDA, where each stream is a queue of sequential tasks, but streams can be processed in parallel, **alpaka** provides an implementation of task parallelism.
Events that can be enqueued to the streams enhance this basic task parallelism by allowing synchronization between different streams, devices or host threads.


Data Parallelism
----------------

The main part of **alpaka** is the way it abstracts data parallelism.
The work is divided into a 1 to 3 dimensional grid of uniform threads appropriate to the problem at hand.
The uniform function executed by each of the threads is called a kernel.

The threads are organized hierarchically and can access local memory at each of the hierarchy levels.
All these higher levels are hidden in the internals of the accelerator implementations and their execution order can not be controlled.

The abstraction used extends the CUDA grid-blocks-threads division strategy explained below by further allowing to facilitate vectorization.
This extended *redundant hierarchical parallelism* scheme is discussed in the paper [The Future of Accelerator Programming: Abstraction, Performance or Can We Have Both?](http://dx.doi.org/10.1109/ICPADS.2013.76) ([PDF](http://olab.is.s.u-tokyo.ac.jp/~kamil.rocki/rocki_burtscher_sac14.pdf)).

### Thread Hierarchy

#### Grid

The whole grid consists of uniform threads each executing the same kernel.
By default threads do not have a cheap way to communicate safely within the whole grid.
This forces decoupling of threads and avoids global interaction and global dependencies.
This independence allows scattering of work blocks within the grid of worker threads and their independent execution on separate processing units. 
Utilizing this property on the higher level allows applications to scale very well.
All threads within the grid can access a global memory.

#### Block

A block is a group of threads.
The whole grid is subdivided into equal sized blocks.
Threads within a block can synchronize and have a fast but small shared memory.
This allows for fast interaction on a local scale.

**TODO**: Why blocks?

#### Thread

Each thread executes the same kernel.
The only difference is the index into the grid which allows each thread to compute a different part of the solution.

**TODO**: more

#### Vectorization (SIMD)

To use the maximum available computing power of a x86 core the computation has to exploit the vector registers.

Because of the x86 SIMD intrinsics (`<xmmintrin.h>`) not being portable, we have to rely on the loop vectorization capabilities of the compiler.

The best solution to vectorization would be one, where the user does not have to do anything.
This is not possible because the smallest unit the user supplies is a kernel which is executed in threads which can synchronize.
It is not possible to hide the vectorization by starting a kernel-thread for e.g. each 4th thread in a block and then looping over the 4 entries.
This would prohibit the synchronization between these threads.
By executing 4 fibers inside such a vectorization kernel-thread we would allow synchronization again but prevent the loop vectorizer from working.

The only possible usage of vectorization is one where we create a kernel-thread for e.g. each 4th thread in a block but do not loop over the 4 threads ourself but rely on the user to implement loops that can be vectorized safely.

### Memory Hierarchy

#### Global Memory

The global memory can be accessed from every thread executing on an accelerator.
This is typically the largest but also the slowest memory available.

#### Shared Memory

Each block has its own shared memory.
This memory can only be accessed by threads within the same block and gets discarded after the complete block finished its calculation.
This memory is typically very fast but also very small.
Sharing has to be done explicitly.
No variables are shared between kernels by default.

#### Registers

This memory is local to each thread.
All variables with default scope defined inside a kernel are automatically saved in registers and not shared automatically.

**TODO**: Constant Memory, Texture Memory?

Mapping *Redundant Hierarchical Parallelism* onto Hardware
--------------------------------------------------------

By providing an accelerator independent interface for kernels, their execution and memory access at different hierarchy levels **alpaka** allows the user to write accelerator independent code that does not neglect performance.

The hard part, the mapping of the decomposition to the execution environment is handled by the **alpaka** library.
The decomposition of the computation in use can not be mapped one to one to any existing hardware.
GPUs do not have vector registers where multiple values of type `int` or `float` can be manipulated by one instruction.
Newer versions of CUDA only implement basic SIMD instructions "on pairs of 16-bit values and quads of 8-bit values". 
They are described in the documentation of the [NVIDIA GPU instruction set architecture](http://docs.nvidia.com/cuda/pdf/ptx_isa_3.2.pdf) chapter 8.7.13 but are only of any use in very special problem domains.
So the vector level is omitted on the CUDA accelerator.
CPUs in turn are not (currently) capable of running thousands of threads concurrently.
Furthermore CPUs do not have an equivalently fast inter-thread synchronization and shared memory access as GPUs do.

A major point of the *redundant hierarchical parallelism* abstraction is to ignore specific unsupported levels and utilize only the ones supported on a specific accelerator.
Furthermore the hierarchy allows a mapping to various current and future accelerators in a variety of ways enabling optimal usage of the underlying compute and memory capabilities.
