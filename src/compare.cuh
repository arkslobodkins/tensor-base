// Arkadijs Slobodkins
// 2024

#pragma once

#include <type_traits>

#include "derived.cuh"


namespace tnb {


template <typename TT1, typename TT2, std::enable_if_t<!TT1::is_device() && !TT2::is_device(), bool> = true>
__host__ bool operator==(const TT1& A, const TT2& B) {
   static_assert(same_value_type<TT1, TT2>());
   static_assert(same_memory_kind<TT1, TT2>());

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
template <typename TT1, typename TT2, std::enable_if_t<TT1::is_device() && TT2::is_device(), bool> = true>
__host__ bool operator==(const TT1& A, const TT2& B) {
   Tensor<ValueTypeOf<TT1>, A.dimension()> Ahat(A.extents());
   Tensor<ValueTypeOf<TT2>, B.dimension()> Bhat(B.extents());
   Ahat.copy_sync(A);
   Bhat.copy_sync(B);
   return Ahat == Bhat;
}


template <typename TT1, typename TT2>
__host__ bool operator!=(const TT1& A, const TT2& B) {
   return !(A == B);
}


}  // namespace tnb
