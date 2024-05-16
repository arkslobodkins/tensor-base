// Arkadijs Slobodkins
// 2024

#pragma once

#include <curand.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

#include "base.cuh"


#define ASSERT_CURAND(curandCall)                                                        \
   do {                                                                                  \
      curandStatus_t error = curandCall;                                                 \
      if(error != CURAND_STATUS_SUCCESS) {                                               \
         std::fprintf(stderr, "CURAND error on line %i, file %s\n", __LINE__, __FILE__); \
         std::exit(EXIT_FAILURE);                                                        \
      }                                                                                  \
   } while(0)


namespace tnb {


namespace internal {
inline auto get_seed() {
   using namespace std::chrono;
   auto duration = system_clock::now().time_since_epoch();
   return duration_cast<nanoseconds>(duration).count();
}


template <typename Gen, typename TensorType>
void rand_uniform(Gen& gen, TensorType& A) {
   using T = ValueTypeOf<TensorType>;
   static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
   if constexpr(std::is_same_v<T, float>) {
      ASSERT_CURAND(curandGenerateUniform(gen, A.data(), A.size()));
   } else {
      ASSERT_CURAND(curandGenerateUniformDouble(gen, A.data(), A.size()));
   }
}


}  // namespace internal


template <typename TensorType>
void random(TensorType& A) {
   auto seed = internal::get_seed();
   curandGenerator_t gen;

   if constexpr(TensorType::host_type()) {
      ASSERT_CURAND(curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT));
   } else {  // device or unified
      ASSERT_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
   }
   ASSERT_CURAND(curandSetPseudoRandomGeneratorSeed(gen, seed));
   internal::rand_uniform(gen, A);
   ASSERT_CURAND(curandDestroyGenerator(gen));
}


}  // namespace tnb
