// Arkadijs Slobodkins
// 2024

#pragma once

#include <cassert>
#include <type_traits>
#include <utility>

#include "base.cuh"

namespace tnb {


namespace internal {


////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, index_t dim, Scheme scheme, bool is_const_ptr, bool is_pinned_mem>
class TensorSliceBase : public LinearBase<T, dim, scheme, is_const_ptr, is_pinned_mem> {
public:
   __host__ __device__ explicit TensorSliceBase(
       std::conditional_t<is_const_ptr, const T*, T*> data, const Extents<dim>& ext) {
      TENSOR_VALIDATE_HOST_DEBUG;
      assert(valid_extents(ext));
      data_ = data;
      ext_ = ext;
   }

   __host__ __device__ TensorSliceBase(const TensorSliceBase& A) {
      TENSOR_VALIDATE_HOST_DEBUG;
      data_ = A.data_;
      ext_ = A.ext_;
   }

   __host__ __device__ TensorSliceBase& operator=(const TensorSliceBase&) = delete;

private:
   using LinearBase<T, dim, scheme, is_const_ptr, is_pinned_mem>::ext_;
   using LinearBase<T, dim, scheme, is_const_ptr, is_pinned_mem>::data_;
};


template <typename T, index_t dim, bool is_const_ptr>
class UnifiedTensorSliceBase : public LinearBaseCommon<T, dim, Unified, is_const_ptr> {
public:
   __host__ __device__ explicit UnifiedTensorSliceBase(
       std::conditional_t<is_const_ptr, const T*, T*> data, const Extents<dim>& ext) {
      assert(valid_extents(ext));
      data_ = data;
      ext_ = ext;
   }

   __host__ __device__ UnifiedTensorSliceBase(const UnifiedTensorSliceBase& A) {
      data_ = A.data_;
      ext_ = A.ext_;
   }

   __host__ __device__ UnifiedTensorSliceBase& operator=(const UnifiedTensorSliceBase&)
       = delete;

private:
   using LinearBaseCommon<T, dim, Unified, is_const_ptr>::ext_;
   using LinearBaseCommon<T, dim, Unified, is_const_ptr>::data_;
};


////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, index_t dim, bool is_pinned_mem>
class TensorSlice : public TensorSliceBase<T, dim, Host, false, is_pinned_mem> {
public:
   using TensorSliceBase<T, dim, Host, false, is_pinned_mem>::TensorSliceBase;
};


template <typename T, index_t dim>
class CudaTensorSlice : public TensorSliceBase<T, dim, Device, false, false> {
public:
   using TensorSliceBase<T, dim, Device, false, false>::TensorSliceBase;

   __host__ [[nodiscard]] auto pass() const {
      return *this;
   }
};


template <typename T, index_t dim, bool is_pinned_mem>
class ConstTensorSlice : public TensorSliceBase<T, dim, Host, true, is_pinned_mem> {
public:
   using TensorSliceBase<T, dim, Host, true, is_pinned_mem>::TensorSliceBase;
};


template <typename T, index_t dim>
class ConstCudaTensorSlice : public TensorSliceBase<T, dim, Device, true, false> {
public:
   using TensorSliceBase<T, dim, Device, true, false>::TensorSliceBase;

   __host__ [[nodiscard]] auto pass() const {
      return *this;
   }
};


////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, index_t dim>
class UnifiedTensorSlice : public UnifiedTensorSliceBase<T, dim, false> {
public:
   using UnifiedTensorSliceBase<T, dim, false>::UnifiedTensorSliceBase;

   __host__ [[nodiscard]] auto pass() const {
      return *this;
   }
};


template <typename T, index_t dim>
class UnifiedConstTensorSlice : public UnifiedTensorSliceBase<T, dim, true> {
public:
   using UnifiedTensorSliceBase<T, dim, true>::UnifiedTensorSliceBase;

   __host__ [[nodiscard]] auto pass() const {
      return *this;
   }
};


template <typename TT, index_t in_dim>
__host__ __device__ auto slice_data(TT&& A, const Extents<in_dim>& ext, index_t offset) {
   using T = typename std::remove_reference_t<TT>::value_type;
   if constexpr(std::is_const_v<std::remove_reference_t<TT>>) {
      if constexpr(A.is_host()) {
         return ConstTensorSlice<T, in_dim, A.is_pinned()>{A.data() + offset, ext};
      } else if constexpr(A.is_device()) {
         return ConstCudaTensorSlice<T, in_dim>{A.data() + offset, ext};
      } else {
         return UnifiedConstTensorSlice<T, in_dim>{A.data() + offset, ext};
      }

   } else {
      if constexpr(A.is_host()) {
         return TensorSlice<T, in_dim, A.is_pinned()>{A.data() + offset, ext};
      } else if constexpr(A.is_device()) {
         return CudaTensorSlice<T, in_dim>{A.data() + offset, ext};
      } else {
         return UnifiedTensorSlice<T, in_dim>{A.data() + offset, ext};
      }
   }
}


}  // namespace internal


template <typename TT, typename... Ints>
__host__ __device__ auto lslice(TT&& A, Ints... indexes) {
   static_assert(std::is_lvalue_reference_v<TT>);
   constexpr auto out_dim = internal::sizeof_cast<Ints...>();
   constexpr auto in_dim = A.dimension() - out_dim;
   static_assert(out_dim < A.dimension());
   static_assert(in_dim > 0);

   Extents<in_dim> sub_ext;
   for(index_t d = 0; d < in_dim; ++d) {
      sub_ext[d] = A.extent(d + out_dim);
   }

   auto offset = A.template offset_of<out_dim, Ints...>(indexes...);
   return internal::slice_data(std::forward<TT>(A), sub_ext, offset);
}


template <typename TT>
__host__ __device__ auto lblock(TT&& A, index_t first, index_t last) {
   static_assert(std::is_lvalue_reference_v<TT>);
   assert(last >= first);
   assert(A.valid_index(first, 0) && A.valid_index(last, 0));

   constexpr auto dim = A.dimension();
   Extents<dim> sub_ext = A.extents();
   sub_ext[0] = last - first + 1;

   auto offset = A.template offset_of<dim>(first);
   return internal::slice_data(std::forward<TT>(A), sub_ext, offset);
}


template <typename TT>
__host__ __device__ auto lblock(TT&& A, index_t first) {
   return lblock(std::forward<TT>(A), first, first);
}


////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, index_t dim>
__host__ [[nodiscard]] auto attach_host(T* data, const Extents<dim>& ext) {
   if constexpr(std::is_const_v<std::remove_pointer_t<T>>) {
      return internal::ConstTensorSlice<T, dim, true>{data, ext};
   } else {
      return internal::TensorSlice<T, dim, true>{data, ext};
   }
}


template <typename T, index_t dim>
__host__ __device__ [[nodiscard]] auto attach_device(T* data, const Extents<dim>& ext) {
   if constexpr(std::is_const_v<std::remove_pointer_t<T>>) {
      return internal::ConstCudaTensorSlice<T, dim>{data, ext};
   } else {
      return internal::CudaTensorSlice<T, dim>{data, ext};
   }
}


}  // namespace tnb

