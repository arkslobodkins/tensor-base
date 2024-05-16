#pragma once

#include "../src/tensor.cuh"

using namespace tnb;


template <typename TensorType, index_t... I>
void construct_tensor() {
   TensorType A(I...);
   random(A);

   TensorType B = A;
   assert(A == B);

   TensorType C = std::move(B);
   assert(B.empty());
   assert(A == C);
}


template <typename T>
void construct() {
   constexpr index_t m = 200;
   constexpr index_t n = 100;
   construct_tensor<Vector<T>, m>();
   construct_tensor<CudaVector<T>, m>();
   construct_tensor<UnifiedVector<T>, m>();

   construct_tensor<Matrix<T>, m, n>();
   construct_tensor<CudaMatrix<T>, m, n>();
   construct_tensor<UnifiedMatrix<T>, m, n>();
}

