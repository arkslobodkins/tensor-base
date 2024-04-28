#include <cassert>
#include <cstdlib>
#include <iostream>
#include <numeric>

#include "../src/tensor.cuh"

using namespace tnb;


template <typename T>
__global__ void compute(const CudaTensor<T, 4> A, const CudaTensor<T, 4> B, CudaTensor<T, 4> C) {
   assert(same_extents(A, B));
   assert(same_extents(B, C));

   index_t i = blockDim.x * blockIdx.x + threadIdx.x;
   for(; i < A.extent(0); i += gridDim.x * blockDim.x)
      for(index_t j = 0; j < C.extent(1); ++j)
         for(index_t k = 0; k < C.extent(2); ++k)
            for(index_t l = 0; l < C.extent(3); ++l)
               C(i, j, k, l) = 2 * A(i, j, k, l) + 2 * B(i, j, k, l);
}


template <typename TensorType>
__global__ void scale(TensorType A) {
   index_t i = blockDim.x * blockIdx.x + threadIdx.x;
   for(; i < A.extent(0); i += gridDim.x * blockDim.x) {
      auto rt = slice(A, i);
      for(auto& x : rt) {
         x *= 1000;
      }
   }
}


int main() {
   const auto ext = Extents<4>(8, 2, 2, 2);
   Tensor<int, 4> A(ext), B(ext);
   CudaTensor<int, 4> A_gpu, B_gpu, C_gpu;
   A_gpu.Allocate(ext);
   B_gpu.Allocate(ext);
   C_gpu.Allocate(ext);

   std::iota(A.begin(), A.end(), 0);
   std::iota(B.begin(), B.end(), 0);

   A_gpu.copy_sync(A);
   B_gpu.copy_sync(B);

   compute<<<2, 2>>>(A_gpu, B_gpu, C_gpu);
   scale<<<2, 2>>>(block(C_gpu, 0, 3));

   A.copy_sync(C_gpu);

   A_gpu.Free();
   B_gpu.Free();
   C_gpu.Free();

   std::cout << A << std::endl;

   return EXIT_SUCCESS;
}
