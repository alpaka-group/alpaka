Threading
=========

There are multiple possible ways to implement threading on non CUDA accelerators.


Serial
------

### Positive
* Easy implementation (no sync).

### Negative
* Restricts block size to 1*1*1. Due to the threads in a block being able to synchronize they have to be executed in parallel. This can not be faked.


OpenMP
------

### Positive
* Lightweight threads.

### Negative
* Interaction with std::threads/pthreads unspecified.
* Thread count limit platform dependent (some runtimes allow oversubscription). No correct way to read the maximum number of threads supported.
* Hard to control thread affinity.
* Non-deterministical thread change.


Kernel-Threads
--------------

std::thread, pthread, ...

### Positive
* Thread affinity is controllable (platform dependent implementation).

### Negative
* High cost of thread creation and thread change.
* Non-deterministical thread change.


Fibers
------

A fiber is a user-space thread with cooperative context-switch.
They are implemented on top of coroutines. A coroutine is a function that can be suspended and resumed but has not necessarily a stack.
boost::fiber = stackful coroutine + scheduler + sync (no wait → next fiber in thread)
C++17: N3858, N3985, (N4134 stackless coroutines 'await')

### Positive
* Less cost of creation.
* Less cost of thread change.
* Deterministic thread change.
* No locks at all (Always only one active fiber per kernel-thread).
* Prevents false sharing because all fibers working on nearby values are in the same block and can be executed by the same kernel-thread on the same core.
* Prevents cache thrashing (threads on the same core compete for the same cache line) by using a user-level scheduler for the fibers that can invocate them in order of access of the memory.
