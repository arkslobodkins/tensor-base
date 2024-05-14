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


#define ASSERT_CURAND(curandCall)                                                        \
   do {                                                                                  \
      curandStatus_t error = curandCall;                                                 \
      if(error != CURAND_STATUS_SUCCESS) {                                               \
         std::fprintf(stderr, "CURAND error on line %i, file %s\n", __LINE__, __FILE__); \
         std::exit(EXIT_FAILURE);                                                        \
      }                                                                                  \
   } while(0)


namespace tnb {


////////////////////////////////////////////////////////////////////////////////////////////////
enum Allocator { Regular, Pinned };


template <typename T, index_t dim, Allocator alloc = Regular>
class Tensor;


template <typename T, index_t dim>
class CudaTensor;


////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, Allocator alloc = Regular>
using Vector = Tensor<T, 1, alloc>;


template <typename T>
using CudaVector = CudaTensor<T, 1>;


template <typename T, Allocator alloc = Regular>
using Matrix = Tensor<T, 2, alloc>;


template <typename T>
using CudaMatrix = CudaTensor<T, 2>;


template <typename T, index_t dim, Allocator alloc>
class Tensor : public LinearBase<T, dim, host> {
public:
   explicit Tensor() = default;


   explicit Tensor(const Extents<dim>& ext) {
      assert(valid_extents(ext));
      ext_ = ext;
      Allocate(ext);
   }


   template <typename... Ints, std::enable_if_t<(... && is_actually_integer<Ints>()), bool> = true>
   explicit Tensor(Ints... ext) : Tensor{Extents<dim>{ext...}} {
   }


   Tensor(const Tensor& A) : Tensor{A.extents()} {
      std::copy(A.begin(), A.end(), this->begin());
   }


   Tensor(Tensor&& A) noexcept {
      this->swap(A);
      A->swap(Tensor{});
   }


   Tensor& operator=(const Tensor& A) {
      assert(same_extents(*this, A));
      if(this != &A) {
         std::copy(A.begin(), A.end(), this->begin());
      }
      return *this;
   }


   Tensor& operator=(Tensor&& A) noexcept {
      assert(same_extents(*this, A));
      if(this != &A) {
         this->swap(A);
         A->swap(Tensor{});
      }
      return *this;
   }


   ~Tensor() {
      this->Free();
   }


   void resize(const Extents<dim>& ext) {
      assert(valid_extents(ext));
      this->Free();
      Allocate(ext);
      ext_ = ext;
   }


   void swap(Tensor& A) noexcept {
      ext_ = std::exchange(A.ext_, ext_);
      std::swap(data_, A.data_);
   }


   void swap(Tensor&& A) noexcept {
      this->swap(A);
   }


private:
   using LinearBase<T, dim, host>::ext_;
   using LinearBase<T, dim, host>::data_;


   void Allocate(const Extents<dim>& ext) {
      if constexpr(alloc == Regular) {
         if(ext.size()) {
            data_ = new T[ext.size()];
         }
      } else {
         if(ext.size()) {
            ASSERT_CUDA(cudaMallocHost(&data_, sizeof(T) * ext.size()));
         }
      }
   }


   void Free() {
      if constexpr(alloc == Regular) {
         delete[] data_;
      } else {
         ASSERT_CUDA(cudaFreeHost(data_));
      }
      data_ = nullptr;
   }
};


template <typename T, index_t dim>
class CudaTensor : public LinearBase<T, dim, device> {
public:
   __host__ explicit CudaTensor() {
   }


   __host__ explicit CudaTensor(const Extents<dim>& ext) {
      assert(valid_extents(ext));
      ext_ = {ext};
      Allocate(ext);
      ASSERT_CUDA(cudaMalloc(&cuda_ptr_, sizeof(CudaTensor)));
      ASSERT_CUDA(cudaMemcpy(cuda_ptr_, this, sizeof(CudaTensor), cudaMemcpyHostToDevice));
   }


   template <typename... Ints>
   __host__ explicit CudaTensor(Ints... ext) : CudaTensor(Extents<dim>{ext...}) {
   }


   __host__ CudaTensor(const CudaTensor& A) : CudaTensor(A.extents()) {
      this->copy_sync(A);
   }


   __host__ CudaTensor(CudaTensor&& A) noexcept : CudaTensor(Extents<dim>{}) {
      this->swap(A);
   }


   __host__ CudaTensor& operator=(const CudaTensor& A) {
      assert(same_extents(*this, A));
      if(this != &A) {
         this->copy_sync(A);
      }
      return *this;
   }


   __host__ CudaTensor& operator=(CudaTensor&& A) noexcept {
      assert(same_extents(*this, A));
      if(this != &A) {
         this->swap(A);
         A->swap(CudaTensor{});
      }
   }


   __host__ ~CudaTensor() {
      ASSERT_CUDA(cudaFree(data_));
      ASSERT_CUDA(cudaFree(cuda_ptr_));
   }


   __host__ void resize(const Extents<dim>& ext) {
      assert(valid_extents(ext));
      ASSERT_CUDA(cudaFree(data_));
      data_ = nullptr;  // avoid double free if ext is zeros

      Allocate(ext);
      ext_ = ext;
   }


   __host__ [[nodiscard]] auto* cuda_ptr() {
      return cuda_ptr_;
   }


   __host__ [[nodiscard]] const auto* cuda_ptr() const {
      return cuda_ptr_;
   }


   __host__ [[nodiscard]] auto* pass() {
      return cuda_ptr();
   }


   __host__ [[nodiscard]] const auto* pass() const {
      return cuda_ptr();
   }


   __host__ void swap(CudaTensor& A) noexcept {
      ext_ = std::exchange(A.ext_, ext_);
      std::swap(data_, A.data_);
   }


   __host__ void swap(CudaTensor&& A) noexcept {
      this->swap(A);
   }


private:
   using LinearBase<T, dim, device>::ext_;
   using LinearBase<T, dim, device>::data_;
   CudaTensor* cuda_ptr_{};

   void Allocate(const Extents<dim>& ext) {
      if(ext.size()) {
         ASSERT_CUDA(cudaMalloc(&data_, this->size() * sizeof(T)));
      }
   }
};


template <typename T, index_t dim>
class UnifiedTensor : public LinearBaseCommon<T, dim, unified> {
public:
   __host__ explicit UnifiedTensor() {
   }


   __host__ explicit UnifiedTensor(const Extents<dim>& ext) {
      assert(valid_extents(ext));
      ext_ = ext;
      Allocate(ext);
   }


   template <typename... Ints, std::enable_if_t<(... && is_actually_integer<Ints>()), bool> = true>
   __host__ explicit UnifiedTensor(Ints... ext) : UnifiedTensor{Extents<dim>{ext...}} {
   }


   __host__ UnifiedTensor(const UnifiedTensor& A) : UnifiedTensor{A.extents()} {
      std::copy(A.begin(), A.end(), this->begin());
   }


   __host__ UnifiedTensor(UnifiedTensor&& A) noexcept {
      this->swap(A);
      A->swap(UnifiedTensor{});
   }


   __host__ UnifiedTensor& operator=(const UnifiedTensor& A) {
      assert(same_extents(*this, A));
      if(this != &A) {
         std::copy(A.begin(), A.end(), this->begin());
      }
      return *this;
   }


   __host__ UnifiedTensor& operator=(UnifiedTensor&& A) noexcept {
      assert(same_extents(*this, A));
      if(this != &A) {
         this->swap(A);
         A->swap(UnifiedTensor{});
      }
      return *this;
   }


   __host__ ~UnifiedTensor() {
      this->Free();
   }


   __host__ void resize(const Extents<dim>& ext) {
      assert(valid_extents(ext));
      this->Free();
      Allocate(ext);
      ext_ = ext;
   }


   __host__ void swap(UnifiedTensor& A) noexcept {
      ext_ = std::exchange(A.ext_, ext_);
      std::swap(data_, A.data_);
   }


   __host__ void swap(UnifiedTensor&& A) noexcept {
      this->swap(A);
   }


private:
   using LinearBaseCommon<T, dim, unified>::ext_;
   using LinearBaseCommon<T, dim, unified>::data_;


   __host__ void Allocate(const Extents<dim>& ext) {
      if(ext.size()) {
         ASSERT_CUDA(cudaMallocManaged(&data_, sizeof(T) * ext.size()));
      }
   }


   __host__ void Free() {
      ASSERT_CUDA(cudaFree(data_));
      data_ = nullptr;
   }
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
   } else {
      ASSERT_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
   }
   ASSERT_CURAND(curandSetPseudoRandomGeneratorSeed(gen, seed));
   internal::rand_uniform(gen, A);
   ASSERT_CURAND(curandDestroyGenerator(gen));
}


}  // namespace tnb
