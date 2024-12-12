#include <cstdlib>
#include <tnb.cuh>


int main() {
   using namespace tnb;

   int N = 100;
   Tensor<double, 4> A(N, N, N, N);
   CudaTensor<double, 4> A_gpu(N, N, N, N);
   UnifiedTensor<double, 4> A_unified(N, N, N, N);
   ASSERT_CUDA(cudaMemPrefetchAsync(A_unified.data(), A_unified.bytes(), 0, 0));

   TIME_EXPR(random(A));
   TIME_EXPR(random(A_gpu));
   TIME_EXPR(random(A_unified));

   return EXIT_SUCCESS;
}
