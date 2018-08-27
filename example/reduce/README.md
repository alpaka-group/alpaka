# Reduction

This is a reduction which works with CPU and GPU accelerators. 

A benchmark of this reduction can be found at [alpaka_reduction_benchmark](https://github.com/kloppstock/alpaka_reduction_benchmark).

## File Descriptions

* [alpakaConfig.hpp](./reduce/src/alpakaConfig.hpp): configurations and settings specific to the individual accelerators.
* [iterator.hpp](./reduce/src/iterator.hpp): contains a CPU and a GPU iterator.
* [kernel.hpp](./reduce/src/kernel.hpp): contains the optimized alpaka reduction kernel.
* [reduce.cpp](./reduce/src/reduce.cpp): the main file.
