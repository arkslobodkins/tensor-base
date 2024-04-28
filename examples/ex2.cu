#include <cstdlib>
#include <iostream>

#include "../src/tensor.cuh"

using namespace tnb;


template <typename T>
__global__ void fill(CudaMatrix<T> A) {
   auto rid = blockDim.x * blockIdx.x + threadIdx.x;
   for(; rid < A.extent(0); rid += gridDim.x * blockDim.x) {
      auto row = lslice(A, rid);
      for(auto& x : row) {
         x = rid;
      }
   }
}


int main() {
   CudaMatrix<float> A;
   A.Allocate(16, 8);
   fill<<<4, 4>>>(A);
   std::cout << A << std::endl;
   A.Free();

   return EXIT_SUCCESS;
}
