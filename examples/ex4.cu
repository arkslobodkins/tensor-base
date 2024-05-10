#include <cstdlib>

#include "../src/tensor.cuh"

using namespace tnb;


template <typename TensorType>
__global__ void add_scalar(TensorType A, typename TensorType::value_type scalar) {
   index_t i = blockDim.x * blockIdx.x + threadIdx.x;
   for(; i < A.size(); i += gridDim.x * blockDim.x) {
      A[i] += scalar;
   }
}


template <typename T, index_t M>
bool verify(const T& x, const typename T::value_type (&scalars)[M]) {
   for(index_t i = 0; i < x.extent(0); ++i) {
      auto s = lslice(x, i);
      for(index_t k = 0; k < s.size(); ++k) {
         if(s[k] > scalars[i] + 1 || s[k] < scalars[i]) {
            return false;
         }
      }
   }
   return true;
}


int main() {
   using T = float;
   constexpr int M = 4;
   constexpr int N = 10000;
   const Extents<3> ext(M, N, N);
   const Extents<2> sub_ext(N, N);
   const auto nbytes = ext.size() / M * sizeof(T);

   T* x;
   ASSERT_CUDA(cudaMallocHost(&x, M * nbytes));
   auto tx = attach_host(x, ext);
   random(tx);

   cudaStream_t streams[M];
   T* x_gpu[M]{};
   const T scalars[M]{1, 11, 101, 1001};

   timer t;
   for(int i = 0; i < M; ++i) {
      ASSERT_CUDA(cudaStreamCreate(&streams[i]));
      ASSERT_CUDA(cudaMallocAsync(&x_gpu[i], nbytes, streams[i]));
      ASSERT_CUDA(cudaMemcpyAsync(x_gpu[i], lslice(tx, i).data(), nbytes, cudaMemcpyDefault, streams[i]));

      auto t_gpu = attach_device(x_gpu[i], sub_ext);  // using x_gpu[i] is safe
      add_scalar<<<8, 8, 0, streams[i]>>>(t_gpu, scalars[i]);

      ASSERT_CUDA(cudaMemcpyAsync(lslice(tx, i).data(), x_gpu[i], nbytes, cudaMemcpyDefault, streams[i]));
      ASSERT_CUDA(cudaFreeAsync(x_gpu[i], streams[i]));
   }

   for(int i = 0; i < M; ++i) {
      ASSERT_CUDA(cudaStreamSynchronize(streams[i]));
      ASSERT_CUDA(cudaStreamDestroy(streams[i]));
   }

   assert(verify(tx, scalars));

   if(N <= 8) {
      std::cout << tx << std::endl;
   }
   ASSERT_CUDA(cudaFreeHost(x));
   ASSERT_CUDA(cudaDeviceReset());

   return EXIT_SUCCESS;
}
