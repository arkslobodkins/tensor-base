// Arkadijs Slobodkins
// 2024

#pragma once

#include <iomanip>
#include <iostream>
#include <type_traits>

#include "derived.cuh"


namespace tnb {


template <template <typename, index_t> class TensorType, typename T,
          std::enable_if_t<TensorType<T, 1>::is_host_type(), bool> = true>
std::ostream& operator<<(std::ostream& os, const TensorType<T, 1>& A) {
   os << std::fixed << std::setprecision(8);
   for(index_t i = 0; i < A.extent(0); ++i) {
      os << A(i) << " ";
   }
   std::cout << std::endl;
   return os;
}


template <template <typename, index_t> class TensorType, typename T,
          std::enable_if_t<TensorType<T, 2>::is_host_type(), bool> = true>
std::ostream& operator<<(std::ostream& os, const TensorType<T, 2>& A) {
   os << std::fixed << std::setprecision(8);

   for(index_t i = 0; i < A.extent(0); ++i) {
      for(index_t j = 0; j < A.extent(1); ++j) {
         os << A(i, j) << " ";
      }
      os << std::endl;
   }
   return os;
}


template <template <typename, index_t> class TensorType, typename T,
          std::enable_if_t<TensorType<T, 3>::is_host_type(), bool> = true>
std::ostream& operator<<(std::ostream& os, const TensorType<T, 3>& A) {
   os << std::fixed << std::setprecision(8);

   for(index_t i = 0; i < A.extent(0); ++i) {
      os << "A(" << i << ", :, :) = " << std::endl;
      for(index_t j = 0; j < A.extent(1); ++j) {
         for(index_t k = 0; k < A.extent(2); ++k) {
            os << A(i, j, k) << " ";
         }
         os << std::endl;
      }
      if(i != A.extent(0) - 1) {
         os << std::endl;
      }
   }
   return os;
}


template <template <typename, index_t> class TensorType, typename T,
          std::enable_if_t<TensorType<T, 4>::is_host_type(), bool> = true>
std::ostream& operator<<(std::ostream& os, const TensorType<T, 4>& A) {
   os << std::fixed << std::setprecision(8);

   for(index_t i = 0; i < A.extent(0); ++i) {
      for(index_t j = 0; j < A.extent(1); ++j) {
         os << "A(" << i << ", " << j << ", :, :) = " << std::endl;
         for(index_t k = 0; k < A.extent(2); ++k) {
            for(index_t l = 0; l < A.extent(3); ++l) {
               os << A(i, j, k, l) << " ";
            }
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
          std::enable_if_t<!TensorType<T, dim>::is_host_type(), bool> = true>
std::ostream& operator<<(std::ostream& os, const TensorType<T, dim>& A) {
   Tensor<T, dim> B(A.extents());
   B.copy_sync(A);
   return os << B;
}


}  // namespace tnb

