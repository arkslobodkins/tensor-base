// Arkadijs Slobodkins
// 2024

#pragma once

#include <curand.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <utility>

#include "base.cuh"


#define ASSERT_CURAND_SUCCESS(curandCall)                                                \
   do {                                                                                  \
      curandStatus_t error = curandCall;                                                 \
      if(error != CURAND_STATUS_SUCCESS) {                                               \
         std::fprintf(stderr, "CURAND error on line %i, file %s\n", __LINE__, __FILE__); \
         std::exit(EXIT_FAILURE);                                                        \
      }                                                                                  \
   } while(0)


namespace tnb {


template <typename T, index_t dim>
class Tensor;


template <typename T, index_t dim>
class CudaTensor;


////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
using Vector = Tensor<T, 1>;


template <typename T>
using CudaVector = CudaTensor<T, 1>;


template <typename T>
using Matrix = Tensor<T, 2>;


template <typename T>
using CudaMatrix = CudaTensor<T, 2>;


////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, index_t dim>
class Tensor : public LinearBase<T, dim, true> {
public:
   explicit Tensor(const Extents<dim>& ext) {
      ext_ = ext;
      if(this->size()) {
         data_ = new T[this->size()];
      }
   }


   template <typename... Ints, std::enable_if_t<(... && is_actually_integer<Ints>()), bool> = true>
   explicit Tensor(Ints... ext) : Tensor{Extents<dim>{ext...}} {
   }


   explicit Tensor() = default;


   Tensor(const Tensor& A) : Tensor{A.extents()} {
      std::copy(A.begin(), A.end(), this->begin());
   }


   Tensor(Tensor&& A) noexcept {
      ext_ = std::exchange(A.ext_, Extents<dim>{});
      data_ = std::exchange(A.data_, {});
   }


   Tensor& operator=(const Tensor& A) {
      static_assert(this->dimension() == A.dimension());
      assert(same_extents(*this, A));
      if(this != &A) {
         std::copy(A.begin(), A.end(), this->begin());
      }
      return *this;
   }


   Tensor& operator=(Tensor&& A) noexcept {
      static_assert(this->dimension() == A.dimension());
      assert(same_extents(*this, A));
      if(this != &A) {
         delete[] data_;
         data_ = std::exchange(A.data_, {});
      }
      return *this;
   }


   ~Tensor() {
      delete[] data_;
   }


private:
   using LinearBase<T, dim, true>::ext_;
   using LinearBase<T, dim, true>::data_;
};


// shallow copy semantics
template <typename T, index_t dim>
class CudaTensor : public LinearBase<T, dim, false> {
public:
   // can only be created on the host!
   explicit __host__ CudaTensor() {
   }


   // can be shallow copied on host and device
   CudaTensor(const CudaTensor&) = default;
   CudaTensor& operator=(const CudaTensor&) = delete;


   __host__ void Allocate(const Extents<dim>& ext) {
      ext_ = {ext};
      if(this->size()) {
         ASSERT_CUDA_SUCCESS(cudaMalloc(&data_, this->size() * sizeof(T)));
      }
   }


   template <typename... Ints>
   __host__ void Allocate(Ints... ext) {
      Allocate(Extents<dim>{ext...});
   }


   __host__ void Free() {
      ASSERT_CUDA_SUCCESS(cudaFree(data_));
      ext_ = Extents<dim>{};
      // not resetting data to null
   }


private:
   using LinearBase<T, dim, false>::ext_;
   using LinearBase<T, dim, false>::data_;
};


////////////////////////////////////////////////////////////////////////////////////////////////
namespace internal {
inline auto get_seed() {
   using namespace std::chrono;
   auto duration = system_clock::now().time_since_epoch();
   return duration_cast<nanoseconds>(duration).count();
}


template <typename Gen, typename TensorType>
void rand_uniform(Gen& gen, TensorType& A) {
   using T = typename TensorType::value_type;
   static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
   if constexpr(std::is_same_v<T, float>) {
      ASSERT_CURAND_SUCCESS(curandGenerateUniform(gen, A.data(), A.size()));
   } else {
      ASSERT_CURAND_SUCCESS(curandGenerateUniformDouble(gen, A.data(), A.size()));
   }
}


}  // namespace internal


template <template <typename, index_t> class TensorType, typename T, index_t dim>
void random(TensorType<T, dim>& A) {
   auto seed = internal::get_seed();
   curandGenerator_t gen;

   if constexpr(std::is_same_v<Tensor<T, dim>, TensorType<T, dim>>) {
      ASSERT_CURAND_SUCCESS(curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT));
   } else {
      ASSERT_CURAND_SUCCESS(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
   }
   ASSERT_CURAND_SUCCESS(curandSetPseudoRandomGeneratorSeed(gen, seed));
   internal::rand_uniform(gen, A);
   ASSERT_CURAND_SUCCESS(curandDestroyGenerator(gen));
}


}  // namespace tnb
