#include <cstdlib>
#include <iostream>

#include "../src/tensor.cuh"

using namespace tnb;


int main() {
   UnifiedTensor<float, 2> A(16, 8);
   std::cout << A << std::endl;
   return EXIT_SUCCESS;
}
