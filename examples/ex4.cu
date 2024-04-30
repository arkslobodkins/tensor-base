#include <cstdlib>

#include "../src/tensor.cuh"

int main() {
   tnb::Tensor<int, 4> x(2, 2, 2, 2);
   x.memset_sync(-1);
   for(tnb::index_t i = 0; i < x.size(); ++i) {
      std::cout << x[i] << std::endl;
   }


   return EXIT_SUCCESS;
}
