// Arkadijs Slobodkins
// 2024

#pragma once

#include <iomanip>
#include <iostream>
#include <type_traits>

#include "derived.cuh"


namespace tnb {


template <typename TensorType,
          std::enable_if_t<TensorType::host_type() && (TensorType::dimension() == 1), bool> = true>
std::ostream& operator<<(std::ostream& os, const TensorType& A) {
   os << std::fixed << std::setprecision(7);
   for(index_t i = 0; i < A.extent(0); ++i) {
      os << A(i) << " ";
   }
   os << std::endl;
   return os;
}


template <typename TensorType,
          std::enable_if_t<TensorType::host_type() && (TensorType::dimension() == 2), bool> = true>
std::ostream& operator<<(std::ostream& os, const TensorType& A) {
   for(index_t i = 0; i < A.extent(0); ++i) {
      os << lslice(A, i);
   }
   return os;
}


template <typename TensorType,
          std::enable_if_t<TensorType::host_type() && (TensorType::dimension() == 3), bool> = true>
std::ostream& operator<<(std::ostream& os, const TensorType& A) {
   for(index_t i = 0; i < A.extent(0); ++i) {
      os << "A(" << i << ", :, :) = " << std::endl;
      os << lslice(A, i);
      if(i != A.extent(0) - 1) {
         os << std::endl;
      }
   }
   return os;
}


template <typename TensorType,
          std::enable_if_t<TensorType::host_type() && (TensorType::dimension() == 4), bool> = true>
std::ostream& operator<<(std::ostream& os, const TensorType& A) {
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


template <template <typename, index_t> class TensorType, typename T, index_t dim,
          std::enable_if_t<TensorType<T, dim>::device_type(), bool> = true>
std::ostream& operator<<(std::ostream& os, const TensorType<T, dim>& A) {
   Tensor<T, dim> B(A.extents());
   B.copy_sync(A);
   return os << B;
}


}  // namespace tnb

