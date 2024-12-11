// Arkadijs Slobodkins
// 2024

#pragma once

#include <iostream>
#include <type_traits>

#include "derived.cuh"
#include "slice.cuh"


namespace tnb {


template <typename TT,
          std::enable_if_t<(TT::is_host() || TT::is_unified()) && (TT::dimension() == 1), bool>
          = true>
__host__ std::ostream& operator<<(std::ostream& os, const TT& A) {
   for(index_t i = 0; i < A.extent(0); ++i) {
      os << A(i) << " ";
   }
   os << std::endl;
   return os;
}


template <typename TT,
          std::enable_if_t<(TT::is_host() || TT::is_unified()) && (TT::dimension() == 2), bool>
          = true>
__host__ std::ostream& operator<<(std::ostream& os, const TT& A) {
   for(index_t i = 0; i < A.extent(0); ++i) {
      os << lslice(A, i);
   }
   return os;
}


template <typename TT,
          std::enable_if_t<(TT::is_host() || TT::is_unified()) && (TT::dimension() == 3), bool>
          = true>
__host__ std::ostream& operator<<(std::ostream& os, const TT& A) {
   for(index_t i = 0; i < A.extent(0); ++i) {
      os << "(" << i << ", :, :) = " << std::endl;
      os << lslice(A, i);
      if(i != A.extent(0) - 1) {
         os << std::endl;
      }
   }
   return os;
}


template <typename TT,
          std::enable_if_t<(TT::is_host() || TT::is_unified()) && (TT::dimension() == 4), bool>
          = true>
__host__ std::ostream& operator<<(std::ostream& os, const TT& A) {
   for(index_t i = 0; i < A.extent(0); ++i) {
      for(index_t j = 0; j < A.extent(1); ++j) {
         os << "(" << i << ", " << j << ", :, :) = " << std::endl;
         os << lslice(A, i, j);
         if(j != A.extent(1) - 1) {
            os << std::endl;
         }
      }
      if(i != A.extent(0) - 1) {
         os << std::endl;
      }
   }
   return os;
}


template <typename TT, std::enable_if_t<TT::is_device(), bool> = true>
__host__ std::ostream& operator<<(std::ostream& os, const TT& A) {
   Tensor<ValueTypeOf<TT>, TT::dimension()> A_host(A.extents());
   A_host.copy(A);
   return os << A_host;
}


template <typename TT, typename... TTArgs>
__host__ void print(const TT& A, const TTArgs&... AArgs) {
   if constexpr(sizeof...(TTArgs) > 0) {
      std::cout << A << std::endl;
      print(AArgs...);
   } else {
      std::cout << A;
   }
}


template <typename TT, typename... TTArgs>
__host__ void printn(const TT& A, const TTArgs&... AArgs) {
   print(A, AArgs...);
   std::cout << std::endl;
}


}  // namespace tnb

