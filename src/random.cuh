// Arkadijs Slobodkins
// 2024

#pragma once

#include <curand.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

#include "common.cuh"


#ifndef NDEBUG
#define ASSERT_CURAND(curand_call)                                                       \
   do {                                                                                  \
      curandStatus_t error = curand_call;                                                \
      if(error != CURAND_STATUS_SUCCESS) {                                               \
         std::fprintf(stderr, "CURAND error on line %i, file %s\n", __LINE__, __FILE__); \
         std::exit(EXIT_FAILURE);                                                        \
      }                                                                                  \
   } while(0)
#else
#define ASSERT_CURAND(curand_call) ((void)0)
#endif


namespace tnb {


namespace internal {


inline auto get_seed() {
   using namespace std::chrono;
   auto duration = system_clock::now().time_since_epoch();
   return duration_cast<nanoseconds>(duration).count();
}


template <typename Gen, typename TT>
void rand_uniform(Gen& gen, TT& A) {
   using T = ValueTypeOf<TT>;
   static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
   if constexpr(std::is_same_v<T, float>) {
      ASSERT_CURAND(curandGenerateUniform(gen, A.data(), A.size()));
   } else {
      ASSERT_CURAND(curandGenerateUniformDouble(gen, A.data(), A.size()));
   }
}


}  // namespace internal


template <typename TT>
void random(TT&& A) {
   using T = typename std::remove_reference_t<TT>;

   auto seed = internal::get_seed();
   curandGenerator_t gen;

   if constexpr(T::is_host()) {
      ASSERT_CURAND(curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT));
   } else {  // Device or unified.
      ASSERT_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
   }
   ASSERT_CURAND(curandSetPseudoRandomGeneratorSeed(gen, seed));
   internal::rand_uniform(gen, A);
   ASSERT_CURAND(curandDestroyGenerator(gen));
}


}  // namespace tnb
