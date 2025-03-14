#pragma once


#include "../src/tnb.cuh"


using namespace tnb;


template <typename TT>
void swap_tensor(const Extents<2>& ext1, const Extents<2>& ext2) {
   TT A(ext1);
   TT B(ext2);
   random(A);
   random(B);

   auto A_old = A, B_old = B;
   A.swap(B);
   assert(A == B_old);
   assert(B == A_old);
}


template <typename TT>
void swap_tensor_self(const Extents<2>& ext) {
   TT A(ext);
   random(A);

   auto A_old = A;
   A.swap(A);
   assert(A == A_old);
}


void swap() {
   using T = float;
   Extents<2> ext1(100, 100);
   Extents<2> ext2(50, 50);

   swap_tensor_self<Matrix<T>>(ext1);
   swap_tensor_self<CudaMatrix<T>>(ext1);
   swap_tensor_self<UnifiedMatrix<T>>(ext1);

   swap_tensor<Matrix<T>>(ext1, ext2);
   swap_tensor<CudaMatrix<T>>(ext1, ext2);
   swap_tensor<UnifiedMatrix<T>>(ext1, ext2);
}

