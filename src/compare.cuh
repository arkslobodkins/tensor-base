// Arkadijs Slobodkins
// 2024

#pragma once

#include <type_traits>

#include "derived.cuh"


namespace tnb {


template <typename TT1, typename TT2,
          std::enable_if_t<!TT1::is_device() && !TT2::is_device(), bool> = true>
__host__ bool operator==(const TT1& A1, const TT2& A2) {
   static_assert(same_value_type<TT1, TT2>());
   static_assert(same_memory_kind<TT1, TT2>());

   if(!same_extents(A1, A2)) {
      return false;
   }
   for(index_t i = 0; i < A1.size(); ++i) {
      if(A1[i] != A2[i]) {
         return false;
      }
   }
   return true;
}


// Not optimized for efficiency, mainly for testing purposes.
template <typename TT1, typename TT2,
          std::enable_if_t<TT1::is_device() && TT2::is_device(), bool> = true>
__host__ bool operator==(const TT1& A1, const TT2& A2) {
   Tensor<ValueTypeOf<TT1>, A1.dimension()> A1_host(A1.extents());
   Tensor<ValueTypeOf<TT2>, A2.dimension()> A2_host(A2.extents());
   A1_host.copy_sync(A1);
   A2_host.copy_sync(A2);
   return A1_host == A2_host;
}


template <typename TT1, typename TT2>
__host__ bool operator!=(const TT1& A1, const TT2& A2) {
   return !(A1 == A2);
}


}  // namespace tnb
