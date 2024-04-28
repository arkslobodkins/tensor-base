// Arkadijs Slobodkins
// 2024

#pragma once

#include <type_traits>

#include "base.cuh"
#include "derived.cuh"

namespace tnb {


////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, index_t dim, bool is_host_t, bool is_const_ptr>
class TensorSliceBase : public LinearBase<T, dim, is_host_t, is_const_ptr> {
public:
   __host__ __device__ explicit TensorSliceBase(std::conditional_t<is_const_ptr, const T*, T*> data,
                                                const Extents<dim>& ext) {
      this->validate_host_type();
      data_ = data;
      ext_ = ext;
   }

   __host__ __device__ TensorSliceBase(const TensorSliceBase& A) {
      this->validate_host_type();
      data_ = A.data_;
      ext_ = A.ext_;
   }

   __host__ __device__ TensorSliceBase& operator=(const TensorSliceBase&) = delete;

private:
   using LinearBase<T, dim, is_host_t, is_const_ptr>::ext_;
   using LinearBase<T, dim, is_host_t, is_const_ptr>::data_;
};


////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, index_t dim>
class TensorSlice : public TensorSliceBase<T, dim, true, false> {
public:
   using TensorSliceBase<T, dim, true, false>::TensorSliceBase;
};


template <typename T, index_t dim>
class CudaTensorSlice : public TensorSliceBase<T, dim, false, false> {
public:
   using TensorSliceBase<T, dim, false, false>::TensorSliceBase;
};


template <typename T, index_t dim>
class ConstTensorSlice : public TensorSliceBase<T, dim, true, true> {
public:
   using TensorSliceBase<T, dim, true, true>::TensorSliceBase;
};


template <typename T, index_t dim>
class ConstCudaTensorSlice : public TensorSliceBase<T, dim, false, true> {
public:
   using TensorSliceBase<T, dim, false, true>::TensorSliceBase;
};


////////////////////////////////////////////////////////////////////////////////////////////////
template <typename TensorType, typename... Ints>
__host__ __device__ auto slice(TensorType&& A, Ints... indexes) {
   using T = typename std::remove_reference_t<TensorType>::value_type;
   static_assert(std::is_lvalue_reference_v<TensorType>);

   constexpr auto out_dim = static_cast<index_t>(sizeof...(Ints));
   constexpr auto in_dim = A.dimension() - out_dim;
   static_assert(out_dim < A.dimension());
   static_assert(in_dim > 0);

   Extents<in_dim> sub_ext;
   for(index_t d = 0; d < in_dim; ++d) {
      sub_ext[d] = A.extent(d + out_dim);
   }

   auto offset = A.template offset_of<out_dim, Ints...>(indexes...);
   if constexpr(std::is_const_v<std::remove_reference_t<TensorType>>) {
      if constexpr(A.host_type()) {
         return ConstTensorSlice<T, in_dim>(A.data() + offset, sub_ext);
      } else {
         return ConstCudaTensorSlice<T, in_dim>(A.data() + offset, sub_ext);
      }
   } else {
      if constexpr(A.host_type()) {
         return TensorSlice<T, in_dim>(A.data() + offset, sub_ext);
      } else {
         return CudaTensorSlice<T, in_dim>(A.data() + offset, sub_ext);
      }
   }
}


}  // namespace tnb

