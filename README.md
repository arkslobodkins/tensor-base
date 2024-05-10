tensor-base is a light-weight header-only library that provides convenient and uniform
interfaces for N-dimensional arrays(tensors) for CPUs and CUDA GPUs. Main features include
slicing, indexing, iterators, and random number initialization with emphasis on type safety.
For greater flexibility, a user can pass raw pointers to tensors using attach_host and
attach_device interfaces. examples directory demonstrates functionality to get started.
Only row-major format is supported. Requires C++17 support and sufficiently recent version of CUDA.
