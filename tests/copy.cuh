#pragma once

#include <algorithm>

#include "../src/tensor.cuh"

using namespace tnb;


template <typename TensorType>
void memset_tensor_sync(const Extents<2>& ext) {
   TensorType A(ext);
   Tensor<ValueTypeOf<TensorType>, 2> ATest(ext);

   A.memset_sync(-1);
   ATest.copy_sync(A);
   assert(std::all_of(ATest.begin(), ATest.end(), [](auto x) {
      return x == -1;
   }));
}


template <typename TensorType1, typename TensorType2>
void copy_tensor_sync(const Extents<2>& ext) {
   TensorType1 A(ext), A_back(ext);
   TensorType2 B(ext);
   random(A);

   B.copy_sync(A);
   A_back.copy_sync(B);
   assert(A == A_back);
}


template <typename TensorType1, typename TensorType2>
void copy_tensor_async(const Extents<2>& ext) {
   TensorType1 A(ext), A_back(ext);
   TensorType2 B(ext);
   random(A);

   B.copy_async(A);
   A_back.copy_async(B);
   cudaDeviceSynchronize();
   assert(A == A_back);
}


void copy() {
   Extents<2> ext(100, 100);

   memset_tensor_sync<Matrix<int>>(ext);
   memset_tensor_sync<CudaMatrix<int>>(ext);

   using T = float;
   using MH = Matrix<T>;
   using MD = CudaMatrix<T>;
   using MU = UnifiedMatrix<T>;

   copy_tensor_sync<MH, MH>(ext);
   copy_tensor_sync<MH, MD>(ext);
   copy_tensor_sync<MD, MH>(ext);
   copy_tensor_sync<MD, MD>(ext);
   copy_tensor_sync<MU, MU>(ext);

   copy_tensor_async<MH, MD>(ext);
   copy_tensor_async<MD, MH>(ext);
   copy_tensor_async<MD, MD>(ext);
}

