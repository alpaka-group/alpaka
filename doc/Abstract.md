**alpaka** - Abstraction Library for Parallel Kernel Acceleration
=================================================================

**alpaka** defines and implements an abstract hierarchical redundant parallelism model.
It exploits parallelism and memory hierarchies at all levels.
This allows to achieve performance portability across various types of accelerators by ignoring specific unsupported levels and utilizing only the ones supported on a specific accelerator.
All hardware types (multi- and many-core CPUs, GPUs and other accelerators) are treated and can be programmed in the same way.