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
std::ostream& operator<<(std::ostream& os, const TT& A) {
   for(index_t i = 0; i < A.extent(0); ++i) {
      os << A(i) << " ";
   }
   os << std::endl;
   return os;
}


template <typename TT,
          std::enable_if_t<(TT::is_host() || TT::is_unified()) && (TT::dimension() == 2), bool>
          = true>
std::ostream& operator<<(std::ostream& os, const TT& A) {
   for(index_t i = 0; i < A.extent(0); ++i) {
      os << lslice(A, i);
   }
   return os;
}


template <typename TT,
          std::enable_if_t<(TT::is_host() || TT::is_unified()) && (TT::dimension() == 3), bool>
          = true>
std::ostream& operator<<(std::ostream& os, const TT& A) {
   for(index_t i = 0; i < A.extent(0); ++i) {
      os << "A(" << i << ", :, :) = " << std::endl;
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
std::ostream& operator<<(std::ostream& os, const TT& A) {
   for(index_t i = 0; i < A.extent(0); ++i) {
      for(index_t j = 0; j < A.extent(1); ++j) {
         os << "A(" << i << ", " << j << ", :, :) = " << std::endl;
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
std::ostream& operator<<(std::ostream& os, const TT& A) {
   Tensor<ValueTypeOf<TT>, TT::dimension()> A_host(A.extents());
   A_host.copy_sync(A);
   return os << A_host;
}


}  // namespace tnb

