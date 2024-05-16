// Arkadijs Slobodkins
// 2024

#pragma once

#include <type_traits>

#include "derived.cuh"


namespace tnb {


template <typename TensorType1, typename TensorType2,
          std::enable_if_t<!TensorType1::device_type() && !TensorType2::device_type(), bool> = true>
__host__ bool operator==(const TensorType1& A, const TensorType2& B) {
   static_assert(same_value_type<TensorType1, TensorType2>());
   static_assert(same_memory_kind<TensorType1, TensorType2>());

   if(!same_extents(A, B)) {
      return false;
   }
   for(index_t i = 0; i < A.size(); ++i) {
      if(A[i] != B[i]) {
         return false;
      }
   }
   return true;
}


// not optimized for efficiency, mainly for testing purposes
template <typename TensorType1, typename TensorType2,
          std::enable_if_t<TensorType1::device_type() && TensorType2::device_type(), bool> = true>
__host__ bool operator==(const TensorType1& A, const TensorType2& B) {
   Tensor<ValueTypeOf<TensorType1>, A.dimension()> Ahat(A.extents());
   Tensor<ValueTypeOf<TensorType2>, B.dimension()> Bhat(B.extents());
   Ahat.copy_sync(A);
   Bhat.copy_sync(B);
   return Ahat == Bhat;
}


template <typename TensorType1, typename TensorType2>
__host__ bool operator!=(const TensorType1& A, const TensorType2& B) {
   return !(A == B);
}


}  // namespace tnb
