// Arkadijs Slobodkins
// 2024

#pragma once

#include <type_traits>

#include "base.cuh"
#include "derived.cuh"

namespace tnb {


////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, index_t dim, bool is_host_t>
class TensorSliceBase : public LinearBase<T, dim, is_host_t> {
public:
   __host__ __device__ explicit TensorSliceBase(T* data, const Extents<dim>& ext) {
      this->validate_host_type();
      data_ = data;
      ext_ = ext;
   }

   __host__ __device__ TensorSliceBase(const TensorSliceBase& A) : TensorSliceBase{A.data(), A.extents} {
   }

   __host__ __device__ TensorSliceBase& operator=(const TensorSliceBase&) = delete;

private:
   using LinearBase<T, dim, is_host_t>::ext_;
   using LinearBase<T, dim, is_host_t>::data_;
};


template <typename T, index_t dim>
class TensorSlice : public TensorSliceBase<T, dim, true> {
public:
   using TensorSliceBase<T, dim, true>::TensorSliceBase;
};


template <typename T, index_t dim>
class CudaTensorSlice : public TensorSliceBase<T, dim, false> {
public:
   using TensorSliceBase<T, dim, false>::TensorSliceBase;
};


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

   if constexpr(std::is_const_v<std::remove_reference_t<TensorType>>) {
      return 0;  // const slice will be returned
   } else {
      if constexpr(A.is_host_type()) {
         return TensorSlice<T, in_dim>(A.data() + A.template offset_of<out_dim, Ints...>(indexes...),
                                       sub_ext);
      } else {
         return CudaTensorSlice<T, in_dim>(A.data() + A.template offset_of<out_dim, Ints...>(indexes...),
                                           sub_ext);
      }
   }
}


}  // namespace tnb

