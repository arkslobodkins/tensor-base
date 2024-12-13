#include <cstdlib>
#include <iostream>
#include <tnb.cuh>


using namespace tnb;


template <typename TensorType>
__global__ void row_iota(TensorType A) {
   auto rid = blockDim.x * blockIdx.x + threadIdx.x;
   for(; rid < A.extent(0); rid += gridDim.x * blockDim.x) {
      for(auto& x : row(A, rid)) {
         x = rid;
      }
   }
}


int main() {
   UnifiedMatrix<float> A(16, 8);
   row_iota<<<4, 4>>>(A.pass());
   ASSERT_CUDA(cudaDeviceSynchronize());
   for(index_t i = 0; i < A.extent(0); ++i) {
      A(i, 0) = 777;
   }
   std::cout << A << std::endl;
   std::cout << view_as<2>(A, A.extents()) << std::endl;

   return EXIT_SUCCESS;
}
