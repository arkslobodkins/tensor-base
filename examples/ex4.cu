#include <algorithm>
#include <cstdlib>
#include <tnb.cuh>


using namespace tnb;


template <typename TensorType>
__global__ void add_scalar(TensorType A, typename TensorType::value_type scalar) {
   index_t i = blockDim.x * blockIdx.x + threadIdx.x;
   for(; i < A.size(); i += gridDim.x * blockDim.x) {
      A[i] += scalar;
   }
}


template <typename TensorType, index_t M>
__host__ bool verify(const TensorType& x,
                     const typename TensorType::value_type (&scalars)[M]) {
   for(index_t i = 0; i < x.extent(0); ++i) {
      auto within_range = [s = scalars[i]](auto z) { return (z >= s) && (z <= s + 1); };
      auto xi = lslice(x, i);
      return std::all_of(xi.begin(), xi.end(), within_range);
   }
   return true;
}


int main() {
   cudaSetDevice(0);
   // Use scope so that cudaDeviceReset() is called after destructors completed.
   {
      using T = float;
      constexpr int M = 4;
      constexpr int N = 10000;
      constexpr Extents<3> ext(M, N, N);
      constexpr Extents<2> sub_ext(N, N);

      Tensor<T, 3, Pinned> x(ext);
      random(x);

      cudaStream_t streams[M];
      constexpr T scalars[M]{1, 11, 101, 1001};
      T* xd[M]{};

      for(int i = 0; i < M; ++i) {
         ASSERT_CUDA(cudaStreamCreate(&streams[i]));
         ASSERT_CUDA(cudaMallocAsync(&xd[i], sub_ext.size() * sizeof(T), streams[i]));
         auto x_gpu = attach_device(xd[i], sub_ext);

         x_gpu.copy_async(lslice(x, i), streams[i]);
         // Launch with few threads and blocks to notice the benefit of streams.
         add_scalar<<<8, 8, 0, streams[i]>>>(x_gpu, scalars[i]);
         lslice(x, i).copy_async(x_gpu, streams[i]);

         ASSERT_CUDA(cudaFreeAsync(xd[i], streams[i]));
      }

      for(auto & stream : streams) {
         ASSERT_CUDA(cudaStreamSynchronize(stream));
         ASSERT_CUDA(cudaStreamDestroy(stream));
      }

      assert(verify(x, scalars));

      if(N <= 8) {
         std::cout << x << std::endl;
      }
   }
   ASSERT_CUDA(cudaDeviceReset());

   return EXIT_SUCCESS;
}
