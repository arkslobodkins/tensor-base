#include <cstdlib>

#include "../src/tensor.cuh"

using namespace tnb;


template <typename TensorType>
__global__ void scale(TensorType A, typename TensorType::value_type scalar) {
   index_t i = blockDim.x * blockIdx.x + threadIdx.x;
   for(; i < A.size(); i += gridDim.x * blockDim.x) {
      A[i] *= scalar;
   }
}


int main() {
   using T = float;
   constexpr int M = 3;
   const Extents<3> ext(M, 4, 4);
   const Extents<2> sub_ext(4, 4);
   const auto nbytes = ext.size() / M * sizeof(T);

   T *x;
   ASSERT_CUDA(cudaMallocHost(&x, M * nbytes));
   auto tx = attach_host(x, ext);
   random(tx);

   cudaStream_t streams[M];
   T *x_gpu[M];
   const T scalars[M]{0.1, 10, 1000};

   for(int i = 0; i < M; ++i) {
      ASSERT_CUDA(cudaStreamCreate(&streams[i]));
   }

   for(int i = 0; i < M; ++i) {
      ASSERT_CUDA(cudaMallocAsync(&x_gpu[i], nbytes, streams[i]));
      ASSERT_CUDA(cudaMemcpyAsync(x_gpu[i], x + i * sub_ext.size(), nbytes, cudaMemcpyHostToDevice, streams[i]));

      auto t_gpu = attach_device(x_gpu[i], sub_ext);
      scale<<<4, 4, 0, streams[i]>>>(t_gpu, scalars[i]);

      ASSERT_CUDA(cudaMemcpyAsync(x + i * sub_ext.size(), x_gpu[i], nbytes, cudaMemcpyDeviceToHost, streams[i]));
      ASSERT_CUDA(cudaFreeAsync(x_gpu[i], streams[i]));
   }

   for(int i = 0; i < M; ++i) {
      ASSERT_CUDA(cudaStreamSynchronize(streams[i]));
      ASSERT_CUDA(cudaStreamDestroy(streams[i]));
   }

   std::cout << tx << std::endl;
   ASSERT_CUDA(cudaFreeHost(x));
   ASSERT_CUDA(cudaDeviceReset());

   return EXIT_SUCCESS;
}
