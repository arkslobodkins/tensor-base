#pragma once

#include "../src/tensor.cuh"

using namespace tnb;


template <typename TensorType, index_t... I>
void resize_tensor() {
   TensorType A;
   A.resize(I...);
   assert(A.extents() == Extents<TensorType::dimension()>{I...});
}


template <typename T>
void resize() {
   constexpr index_t m = 200;
   constexpr index_t n = 100;
   resize_tensor<Vector<T>, m>();
   resize_tensor<CudaVector<T>, m>();
   resize_tensor<UnifiedVector<T>, m>();

   resize_tensor<Matrix<T>, m, n>();
   resize_tensor<CudaMatrix<T>, m, n>();
   resize_tensor<UnifiedMatrix<T>, m, n>();

   resize_tensor<Matrix<T>, 0, 0>();
   resize_tensor<CudaMatrix<T>, 0, 0>();
   resize_tensor<UnifiedMatrix<T>, 0, 0>();
}

