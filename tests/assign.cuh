#pragma once

#include "../src/tnb.cuh"

using namespace tnb;


template <typename TT, index_t... I>
void assign_tensor() {
   TT A(I...), B(I...), C(I...);
   random(A);

   B = A;
   assert(A == B);

   C = std::move(B);
   assert(A == C);
   assert(B.empty());
}


template <typename T>
void assign() {
   constexpr index_t m = 200;
   constexpr index_t n = 100;

   assign_tensor<Vector<T>, m>();
   assign_tensor<CudaVector<T>, m>();
   assign_tensor<UnifiedVector<T>, m>();

   assign_tensor<Matrix<T>, m, n>();
   assign_tensor<CudaMatrix<T>, m, n>();
   assign_tensor<UnifiedMatrix<T>, m, n>();
}

