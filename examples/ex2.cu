#include <cstdlib>
#include <iostream>

#include "../src/tensor.cuh"

using namespace tnb;


template <typename T>
__global__ void row_iota(CudaMatrix<T>* A) {
   auto rid = blockDim.x * blockIdx.x + threadIdx.x;
   for(; rid < A->extent(0); rid += gridDim.x * blockDim.x) {
      auto row = lslice(*A, rid);
      for(auto& x : row) {
         x = rid;
      }
   }
}


int main() {
   CudaMatrix<float> A(16, 8);
   row_iota<<<4, 4>>>(A.cuda_ptr());
   // cudaDeviceSynchronize is not used since output operator implicitly synchronizes.
   std::cout << A << std::endl;
   return EXIT_SUCCESS;
}
