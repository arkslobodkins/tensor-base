#include <cstdlib>
#include <iostream>

#include "tensor.cuh"
#include "timer.cuh"


int main() {
   using namespace tnb;
   using T = double;

   Tensor<T, 4> x(10, 10, 10, 10);
   CudaTensor<T, 4> x_gpu;
   x_gpu.Allocate(x.extents());

   TIME_EXPR(random(x));
   TIME_EXPR(random(x_gpu));

   x_gpu.Free();

   return EXIT_SUCCESS;
}
