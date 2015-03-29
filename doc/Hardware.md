Hardware
========

Before the abstraction can be defined all possible ways parallelism can be utilised in current and in future hardware has to be considered.
On the one side there are current multi-core multi-socket CPUs and on the other hand there are special many-core accelerators like GPUs, the Intel XeonPhi and many more.

**alpaka** defines an abstract acceleration model that treats and allows to program all of these hardware types in the same way.


GPUs
----

### CUDA

**TODO**


CPUs
----

### x86

**TODO**


CUDA vs x86
-----------

A CUDA capable GPU can be seen as enhanced vector processor.
The following table shows the mapping of CUDA entities to x86 CPUs.

|CUDA|x86|
|---|---|
|GPU|CPU|
|global memory|RAM|
|SM (streaming multiprocessor)|core|
|shared memory|L1&L2 cache|
|block (set of warps on a SM)|simultaneous multi-threading (HyperThreading)|
|(threads per block)/(threads per warp)|#HyperThreads|
|warp (lock-step)|vector register|
|warp size|vector length|

**TODO**: Extend and add source.

### Warps vs. Vector Registers

CUDA executes warps in lock-step.
This means that all threads in a warp execute the same instruction at the same time.
The effect is a simpler GPU hardware design by reducing the amount of hardware control logic required.
Warps are nearly equivalent to vector processors operating on each lane of a vector register simultaneously.
The key difference is that CUDA additionally allows thread divergence due to control flow statements.
All threads in a warp that are in the false branch of a conditional statement are executing no-ops instead of the instructions in the true branch and vice-versa.
Most SIMD implementations on CPUs do not have this capability.
For example an SSE or AVX instruction will unconditionally be executed on all lanes of a vector register.
The only way to imitate it is by using additional masking operations.
See [this example](http://felix.abecassis.me/2012/08/sse-vectorizing-conditional-code/) for computing the square-root of all positive floating point numbers in an array using SSE.
Today`s auto-vectorizers are unable to vectorize such loops with control flow statements inside.