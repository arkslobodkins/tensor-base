// Arkadijs Slobodkins
// 2024


#pragma once


#include <cassert>
#include <type_traits>
#include <utility>

#include "base.cuh"
#include "extents.cuh"


namespace tnb {


namespace internal {


////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, index_t dim, Scheme scheme, bool is_const_ptr, bool is_pinned_mem>
class TensorSliceBase
    : public TensorBaseValidated<T, dim, scheme, is_const_ptr, is_pinned_mem> {
public:
   __host__ __device__ explicit TensorSliceBase(
       std::conditional_t<is_const_ptr, const T*, T*> data, const Extents<dim>& ext) {
      TensorBaseValidated<T, dim, scheme, is_const_ptr, is_pinned_mem>::validate_host_debug();
      assert(valid_extents(ext));
      data_ = data;
      ext_ = ext;
   }

   TensorSliceBase(const TensorSliceBase& A) = default;
   __host__ __device__ TensorSliceBase& operator=(const TensorSliceBase&) = delete;

private:
   using TensorBaseValidated<T, dim, scheme, is_const_ptr, is_pinned_mem>::ext_;
   using TensorBaseValidated<T, dim, scheme, is_const_ptr, is_pinned_mem>::data_;
};


template <typename T, index_t dim, bool is_const_ptr>
class UnifiedTensorSliceBase : public TensorBase<T, dim, Unified, is_const_ptr> {
public:
   __host__ __device__ explicit UnifiedTensorSliceBase(
       std::conditional_t<is_const_ptr, const T*, T*> data, const Extents<dim>& ext) {
      assert(valid_extents(ext));
      data_ = data;
      ext_ = ext;
   }

   UnifiedTensorSliceBase(const UnifiedTensorSliceBase& A) = default;
   __host__ __device__ UnifiedTensorSliceBase& operator=(const UnifiedTensorSliceBase&)
       = delete;

private:
   using TensorBase<T, dim, Unified, is_const_ptr>::ext_;
   using TensorBase<T, dim, Unified, is_const_ptr>::data_;
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
   using value_type = typename std::remove_reference_t<TT>::value_type;
   using T = typename std::remove_reference_t<TT>;

   if constexpr(std::is_const_v<std::remove_reference_t<TT>>) {
      if constexpr(T::is_host()) {
         return ConstTensorSlice<value_type, in_dim, T::is_pinned()>{A.data() + offset, ext};
      } else if constexpr(T::is_device()) {
         return ConstCudaTensorSlice<value_type, in_dim>{A.data() + offset, ext};
      } else {
         return UnifiedConstTensorSlice<value_type, in_dim>{A.data() + offset, ext};
      }

   } else {
      if constexpr(T::is_host()) {
         return TensorSlice<value_type, in_dim, T::is_pinned()>{A.data() + offset, ext};
      } else if constexpr(T::is_device()) {
         return CudaTensorSlice<value_type, in_dim>{A.data() + offset, ext};
      } else {
         return UnifiedTensorSlice<value_type, in_dim>{A.data() + offset, ext};
      }
   }
}


}  // namespace internal


template <typename TT, typename... Ints>
__host__ __device__ auto lslice(TT&& A, Ints... indexes) {
   using T = typename std::decay_t<TT>;
   if constexpr(internal::has_swap<T>::value) {
      static_assert(std::is_lvalue_reference_v<TT>);
   }

   constexpr auto out_dim = internal::sizeof_cast<Ints...>();
   constexpr auto in_dim = T::dimension() - out_dim;
   static_assert(out_dim < T::dimension());
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
   using T = typename std::decay_t<TT>;
   if constexpr(internal::has_swap<T>::value) {
      static_assert(std::is_lvalue_reference_v<TT>);
   }
   assert(last >= first);
   assert(A.valid_index(first, 0) && A.valid_index(last, 0));

   constexpr auto dim = T::dimension();
   Extents<dim> sub_ext = A.extents();
   sub_ext[0] = last - first + 1;

   auto offset = A.template offset_of<dim>(first);
   return internal::slice_data(std::forward<TT>(A), sub_ext, offset);
}


template <typename TT>
__host__ __device__ auto row(TT&& A, index_t i) {
   static_assert(std::decay_t<TT>::dimension() == 2L);
   return lslice(std::forward<TT>(A), i);
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

