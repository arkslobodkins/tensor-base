#pragma once

#include "../src/tensor.cuh"

using namespace tnb;


template <typename TT>
void index_tensor_6d(const Extents<6>& ext) {
   TT A(ext);
   for(auto i = 0, count = 0; i < A.extent(0); ++i)
      for(auto j = 0; j < A.extent(1); ++j)
         for(auto k = 0; k < A.extent(2); ++k)
            for(auto l = 0; l < A.extent(3); ++l)
               for(auto m = 0; m < A.extent(4); ++m)
                  for(auto n = 0; n < A.extent(5); ++n, ++count)
                     assert(A.index_of(i, j, k, l, m, n) == count);
}


template <typename T>
void index() {
   Extents<6> ext1{2, 4, 8, 16, 32, 64};
   index_tensor_6d<Tensor<T, 6>>(ext1);
   index_tensor_6d<CudaTensor<T, 6>>(ext1);
   index_tensor_6d<UnifiedTensor<T, 6>>(ext1);

   Extents<6> ext2{1, 1, 1, 1, 1, 1};
   index_tensor_6d<Tensor<T, 6>>(ext2);
   index_tensor_6d<CudaTensor<T, 6>>(ext2);
   index_tensor_6d<UnifiedTensor<T, 6>>(ext2);
}

