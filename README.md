tensor-base is a light-weight header-only library that provides convenient and
uniform interfaces for arbitrary N-dimensional arrays(tensors) for CPUs and CUDA GPUs.
Main features include contiguous slicing, indexing, iterators, unified memory and
random number initialization with emphasis on type safety. Only row-major format is supported.
For greater flexibility, a user can pass raw pointers to tensors using attach_host and attach_device interfaces.

Requires C++17 and sufficiently recent version of CUDA(11 or 12).
examples directory demonstrates functionality to get started.

