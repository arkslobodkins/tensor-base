#pragma once

#include "../src/tensor.cuh"

using namespace tnb;


template <typename TT, index_t... I>
void construct_tensor() {
   TT A(I...);
   random(A);

   TT B = A;
   assert(A == B);

   TT C = std::move(B);
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

