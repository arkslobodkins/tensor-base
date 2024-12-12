// Arkadijs Slobodkins
// 2024


#pragma once


#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <utility>


#ifndef NDEBUG
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
#else
#define ASSERT_CUDA(cuda_call) ((void)0)
#endif


namespace tnb {


using index_t = long int;


template <typename T>
using ValueTypeOf = typename T::value_type;


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


template <typename T, typename = void>
struct has_swap : std::false_type {};


template <typename T>
struct has_swap<T, std::void_t<decltype(std::declval<T>().swap(std::declval<T>()))>>
    : std::true_type {};


}  // namespace internal


}  // namespace tnb

