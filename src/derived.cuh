// Arkadijs Slobodkins
// 2024


#pragma once


#include <algorithm>
#include <cassert>
#include <type_traits>
#include <utility>

#include "base.cuh"
#include "extents.cuh"
#include "slice.cuh"


namespace tnb {


enum Allocator { Regular, Pinned };


////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, index_t dim, Allocator alloc = Regular>
class Tensor : public TensorBaseValidated<T, dim, Host, false, (alloc == Pinned)> {
public:
   Tensor() = default;


   explicit Tensor(const Extents<dim>& ext) {
      assert(valid_extents(ext));
      ext_ = ext;
      this->Allocate(ext);
   }


   template <typename... Ints>
   explicit Tensor(Ints... ext) : Tensor{Extents<dim>{ext...}} {
   }


   Tensor(const Tensor& A) : Tensor{A.extents()} {
      std::copy(A.begin(), A.end(), this->begin());
   }


   Tensor(Tensor&& A) noexcept {
      this->swap(A);
      A.swap(Tensor{});
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
         A.swap(Tensor{});
      }
      return *this;
   }


   ~Tensor() {
      this->Free();
   }


   void resize(const Extents<dim>& ext) {
      assert(valid_extents(ext));
      this->Free();
      this->Allocate(ext);
      ext_ = ext;
   }


   template <typename... Ints>
   void resize(Ints... ext) {
      this->resize(Extents<dim>{ext...});
   }


   void swap(Tensor& A) noexcept {
      std::swap(ext_, A.ext_);
      std::swap(data_, A.data_);
   }


   void swap(Tensor&& A) noexcept {
      this->swap(A);
   }


   static constexpr auto allocator() {
      return alloc;
   }


private:
   using Base = TensorBaseValidated<T, dim, Host, false, (alloc == Pinned)>;
   using Base::data_;
   using Base::ext_;


   void Allocate(const Extents<dim>& ext) {
      assert(data_ == nullptr);
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


   void Free() noexcept {
      if constexpr(alloc == Regular) {
         delete[] data_;
      } else {
         ASSERT_CUDA(cudaFreeHost(data_));
      }
      data_ = nullptr;
   }
};


////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Base>
class CudaTensorDerived : public Base {
public:
   __host__ CudaTensorDerived() : Base{} {
   }


   __host__ explicit CudaTensorDerived(const Extents<Base::dimension()>& ext)
       : CudaTensorDerived{} {
      assert(valid_extents(ext));
      ext_ = {ext};
      this->Allocate(ext);
   }


   template <typename... Ints>
   __host__ explicit CudaTensorDerived(Ints... ext)
       : CudaTensorDerived{Extents<Base::dimension()>{ext...}} {
   }


   __host__ CudaTensorDerived(const CudaTensorDerived& A) : CudaTensorDerived{A.extents()} {
      ASSERT_CUDA(cudaMemcpy(this->data(), A.data(), this->bytes(), cudaMemcpyDeviceToDevice));
   }


   __host__ CudaTensorDerived(CudaTensorDerived&& A) noexcept
       : CudaTensorDerived{Extents<Base::dimension()>{}} {
      this->swap(A);
   }


   __host__ CudaTensorDerived& operator=(const CudaTensorDerived& A) {
      assert(same_extents(*this, A));
      if(this != &A) {
         ASSERT_CUDA(
             cudaMemcpy(this->data(), A.data(), this->bytes(), cudaMemcpyDeviceToDevice));
      }
      return *this;
   }


   __host__ CudaTensorDerived& operator=(CudaTensorDerived&& A) noexcept {
      assert(same_extents(*this, A));
      if(this != &A) {
         this->swap(A);
         A.swap(CudaTensorDerived{});
      }
      return *this;
   }


   __host__ ~CudaTensorDerived() {
      ASSERT_CUDA(cudaFree(data_));
   }


   __host__ void resize(const Extents<Base::dimension()>& ext) {
      assert(valid_extents(ext));
      ASSERT_CUDA(cudaFree(data_));
      data_ = nullptr;  // Avoid double free if ext is zeros.

      this->Allocate(ext);
      ext_ = ext;
   }


   template <typename... Ints>
   __host__ void resize(Ints... ext) {
      this->resize(Extents<Base::dimension()>{ext...});
   }


   __host__ [[nodiscard]] auto pass() {
      return lblock(*this, 0, ext_[0] - 1L);
   }


   __host__ [[nodiscard]] const auto pass() const {
      return lblock(*this, 0, ext_[0] - 1L);
   }


   __host__ void swap(CudaTensorDerived& A) noexcept {
      std::swap(ext_, A.ext_);
      std::swap(data_, A.data_);
   }


   __host__ void swap(CudaTensorDerived&& A) noexcept {
      this->swap(A);
   }


private:
   using Base::data_;
   using Base::ext_;

   void Allocate(const Extents<Base::dimension()>& ext) {
      assert(data_ == nullptr);
      if(ext.size()) {
         if constexpr(this->is_device()) {
            ASSERT_CUDA(cudaMalloc(&data_, ext.size() * sizeof(ValueTypeOf<Base>)));
         } else if constexpr(this->is_unified()) {
            ASSERT_CUDA(cudaMallocManaged(&data_, ext.size() * sizeof(ValueTypeOf<Base>)));
         } else {
            internal::static_assert_false<Base>();
         }
      }
   }
};


////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, index_t dim>
using CudaTensor = CudaTensorDerived<TensorBaseValidated<T, dim, Device>>;


template <typename T, index_t dim>
using UnifiedTensor = CudaTensorDerived<TensorBase<T, dim, Unified>>;


////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, Allocator alloc = Regular>
using Vector = Tensor<T, 1, alloc>;


template <typename T>
using CudaVector = CudaTensor<T, 1>;


template <typename T>
using UnifiedVector = UnifiedTensor<T, 1>;


////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, Allocator alloc = Regular>
using Matrix = Tensor<T, 2, alloc>;


template <typename T>
using CudaMatrix = CudaTensor<T, 2>;


template <typename T>
using UnifiedMatrix = UnifiedTensor<T, 2>;


}  // namespace tnb
