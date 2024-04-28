#include <cstdlib>

#include "../src/tensor.cuh"

int main() {
   using namespace tnb;

   int N = 100;
   Tensor<double, 4> A(N, N, N, N);
   CudaTensor<double, 4> A_gpu;
   A_gpu.Allocate(A.extents());

   TIME_EXPR(random(A));
   TIME_EXPR(random(A_gpu));

   A_gpu.Free();

   return EXIT_SUCCESS;
}
