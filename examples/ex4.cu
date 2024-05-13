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
   const auto nbytes = sub_ext.size() * sizeof(T);

   Tensor<T, 3, Pinned> x(ext);
   random(x);

   cudaStream_t streams[M];
   T* xd[M]{};
   const T scalars[M]{1, 11, 101, 1001};

   timer t;
   for(int i = 0; i < M; ++i) {
      ASSERT_CUDA(cudaStreamCreate(&streams[i]));
      ASSERT_CUDA(cudaMallocAsync(&xd[i], nbytes, streams[i]));
      auto x_gpu = attach_device(xd[i], sub_ext);  // using xd[i] is safe

      x_gpu.copy_async(lslice(x, i), streams[i]);
      add_scalar<<<8, 8, 0, streams[i]>>>(x_gpu, scalars[i]);
      lslice(x, i).copy_async(x_gpu, streams[i]);

      ASSERT_CUDA(cudaFreeAsync(xd[i], streams[i]));
   }

   for(int i = 0; i < M; ++i) {
      ASSERT_CUDA(cudaStreamSynchronize(streams[i]));
      ASSERT_CUDA(cudaStreamDestroy(streams[i]));
   }

   assert(verify(x, scalars));

   if(N <= 8) {
      std::cout << x << std::endl;
   }
   ASSERT_CUDA(cudaDeviceReset());

   return EXIT_SUCCESS;
}
