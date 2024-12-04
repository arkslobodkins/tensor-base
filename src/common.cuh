// Arkadijs Slobodkins
// 2024

#pragma once


#include <cstdio>
#include <cstdlib>
#include <type_traits>


#define ASSERT_CUDA(cuda_call)                           \
   do {                                                  \
      cudaError_t error = cuda_call;                     \
      if(error != cudaSuccess) {                         \
         std::fprintf(stderr,                            \
                      "Error on line %i, file %s: %s\n", \
                      __LINE__,                          \
                      __FILE__,                          \
                      cudaGetErrorString(error));        \
         std::exit(EXIT_FAILURE);                        \
      }                                                  \
   } while(0)


namespace tnb {


using index_t = long int;


namespace internal {


template <typename... Args>
__host__ __device__ constexpr index_t sizeof_cast() {
   return static_cast<index_t>(sizeof...(Args));
}


template <typename T>
__host__ __device__ constexpr index_t index_cast(T i) {
   return static_cast<index_t>(i);
}


template <typename T>
__host__ __device__ constexpr bool is_compatible_integer() {
   return std::is_same_v<short int, T> || std::is_same_v<unsigned short int, T>
       || std::is_same_v<int, T> || std::is_same_v<unsigned int, T>
       || std::is_same_v<long int, T>;
}


template <typename T>
__host__ __device__ constexpr void static_assert_false() {
   static_assert(!sizeof(T));
}


}  // namespace internal


}  // namespace tnb

