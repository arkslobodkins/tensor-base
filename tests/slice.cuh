#pragma once


#include <memory>
#include <numeric>
#include <type_traits>

#include "../src/tnb.cuh"


using namespace tnb;


template <typename MT, index_t... I>
void slice_tensor_types() {
   MT A(I...);
   auto slice = lslice(A, 0);
   using value_type = ValueTypeOf<MT>;

   static_assert(same_memory_kind<MT, decltype(slice)>());

   static_assert(std::is_same_v<decltype(slice(0)), value_type&>);
   static_assert(std::is_same_v<decltype(slice[0]), value_type&>);

   static_assert(std::is_same_v<decltype(slice.data()), value_type*>);
   static_assert(std::is_same_v<decltype(slice.begin()), value_type*>);
   static_assert(std::is_same_v<decltype(slice.cbegin()), const value_type*>);

   static_assert(std::is_same_v<decltype(slice.end()), value_type*>);
   static_assert(std::is_same_v<decltype(slice.cend()), const value_type*>);

   const auto const_slice = lslice(A, 0);
   static_assert(std::is_same_v<decltype(const_slice(0)), const value_type&>);
   static_assert(std::is_same_v<decltype(const_slice[0]), const value_type&>);

   static_assert(std::is_same_v<decltype(const_slice.data()), const value_type*>);
   static_assert(std::is_same_v<decltype(const_slice.begin()), const value_type*>);
   static_assert(std::is_same_v<decltype(const_slice.cbegin()), const value_type*>);

   static_assert(std::is_same_v<decltype(const_slice.end()), const value_type*>);
   static_assert(std::is_same_v<decltype(const_slice.cend()), const value_type*>);
}


template <typename MT, index_t... I>
void const_slice_tensor_types() {
   const MT A(I...);
   auto slice = lslice(A, 0);
   using value_type = ValueTypeOf<MT>;

   static_assert(std::is_same_v<decltype(slice(0)), const value_type&>);
   static_assert(std::is_same_v<decltype(slice[0]), const value_type&>);

   static_assert(std::is_same_v<decltype(slice.data()), const value_type*>);
   static_assert(std::is_same_v<decltype(slice.begin()), const value_type*>);
   static_assert(std::is_same_v<decltype(slice.cbegin()), const value_type*>);

   static_assert(std::is_same_v<decltype(slice.end()), const value_type*>);
   static_assert(std::is_same_v<decltype(slice.cend()), const value_type*>);
}


void attach_tensor() {
   index_t n = 5 * 10 * 15 * 20;
   std::unique_ptr<int[]> x(new int[n]);
   std::iota(x.get(), x.get() + n, 0);

   auto A = attach_host(x.get(), Extents<4>{5, 10, 15, 20});
   for(index_t i = 0; i < n; ++i) {
      assert(A[i] == x.get()[i]);
   }

   for(index_t i = 0; i < 5; ++i) {
      for(index_t j = 0; j < 10; ++j) {
         for(index_t k = 0; k < 15; ++k) {
            for(index_t l = 0; l < 20; ++l) {
               assert(A(i, j, k, l) == x.get()[A.index_of(i, j, k, l)]);
            }
         }
      }
   }
}


template <typename MT, typename TT>
void lslice_tensor(index_t n) {
   MT M[n];
   for(index_t i = 0; i < n; ++i) {
      M[i].resize(3, 4);
      random(M[i]);
   }

   TT A(n, 3, 4);
   for(index_t i = 0; i < n; ++i) {
      lslice(A, i).copy(M[i]);
      assert(lslice(A, i) == M[i]);
      for(index_t j = 0; j < 3; ++j) {
         auto Ai = lslice(A, i);
         assert(lslice(Ai, j) == lslice(M[i], j));
      }
   }
}


template <typename T>
void slice() {
   slice_tensor_types<Matrix<T>, 10, 10>();
   const_slice_tensor_types<Matrix<T>, 10, 10>();
   slice_tensor_types<CudaMatrix<T>, 10, 10>();
   const_slice_tensor_types<CudaMatrix<T>, 10, 10>();
   slice_tensor_types<UnifiedMatrix<T>, 10, 10>();
   const_slice_tensor_types<UnifiedMatrix<T>, 10, 10>();

   attach_tensor();

   lslice_tensor<Matrix<T>, Tensor<T, 3>>(10);
   lslice_tensor<CudaMatrix<T>, CudaTensor<T, 3>>(10);
   lslice_tensor<UnifiedMatrix<T>, UnifiedTensor<T, 3>>(10);
}

