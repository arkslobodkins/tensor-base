#pragma once


#include <algorithm>

#include "../src/tnb.cuh"


using namespace tnb;


template <typename TT>
void memset_tensor(const Extents<2>& ext) {
   TT A(ext);
   Tensor<ValueTypeOf<TT>, 2> ATest(ext);

   A.memset(-1);
   ATest.copy(A);
   assert(std::all_of(ATest.begin(), ATest.end(), [](auto x) { return x == -1; }));
}


template <typename TT1, typename TT2>
void copy_tensor(const Extents<2>& ext) {
   TT1 A(ext), A_back(ext);
   TT2 B(ext);
   random(A);

   B.copy(A);
   A_back.copy(B);
   assert(A == A_back);
}


template <typename TT1, typename TT2>
void copy_tensor_async(const Extents<2>& ext) {
   TT1 A(ext), A_back(ext);
   TT2 B(ext);
   random(A);

   B.copy_async(A);
   A_back.copy_async(B);
   cudaDeviceSynchronize();
   assert(A == A_back);
}


void copy() {
   Extents<2> ext(100, 100);

   memset_tensor<Matrix<int>>(ext);
   memset_tensor<CudaMatrix<int>>(ext);
   memset_tensor<UnifiedMatrix<int>>(ext);

   using T = float;
   using MH = Matrix<T, Pinned>;
   using MD = CudaMatrix<T>;
   using MU = UnifiedMatrix<T>;

   copy_tensor<MH, MH>(ext);
   copy_tensor<MH, MD>(ext);
   copy_tensor<MD, MH>(ext);
   copy_tensor<MD, MD>(ext);
   copy_tensor<MU, MU>(ext);
   copy_tensor<MU, MH>(ext);
   copy_tensor<MH, MU>(ext);
   copy_tensor<MU, MD>(ext);
   copy_tensor<MD, MU>(ext);

   copy_tensor_async<MH, MD>(ext);
   copy_tensor_async<MD, MH>(ext);
   copy_tensor_async<MD, MD>(ext);
   copy_tensor_async<MU, MU>(ext);
   copy_tensor_async<MU, MH>(ext);
   copy_tensor_async<MH, MU>(ext);
   copy_tensor_async<MU, MD>(ext);
   copy_tensor_async<MD, MU>(ext);
}

