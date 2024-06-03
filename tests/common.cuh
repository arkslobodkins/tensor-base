#pragma once

#include "../src/tensor.cuh"

using namespace tnb;


template <typename TT, index_t... I>
void common_tensor() {
   TT A(I...);

   static_assert(A.dimension() == sizeof...(I));
   assert(A.extents() == Extents<A.dimension()>(I...));

   // the following are trivial and only tested for compilation
   (void)A.data();
   (void)A.empty();
   (void)A.is_host();
   (void)A.is_device();
   (void)A.is_unified();
   (void)A.memory_kind();
   (void)A.size();
   (void)A.bytes();
   (void)A.extent(0);
}


template <typename T>
void common() {
   constexpr index_t m = 200;
   constexpr index_t n = 100;
   common_tensor<Vector<T>, m>();
   common_tensor<CudaVector<T>, m>();
   common_tensor<UnifiedVector<T>, m>();

   common_tensor<Matrix<T>, m, n>();
   common_tensor<CudaMatrix<T>, m, n>();
   common_tensor<UnifiedMatrix<T>, m, n>();

   common_tensor<Matrix<T>, 0, 0>();
   common_tensor<CudaMatrix<T>, 0, 0>();
   common_tensor<UnifiedMatrix<T>, 0, 0>();
}

