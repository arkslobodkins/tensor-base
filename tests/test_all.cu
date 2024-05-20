#include <cstdlib>

#include "assign.cuh"
#include "common.cuh"
#include "construct.cuh"
#include "copy.cuh"
#include "index.cuh"
#include "resize.cuh"
#include "same.cuh"
#include "slice.cuh"
#include "swap.cuh"
#include "test.cuh"
#include "valid.cuh"

int main() {
   TEST_ALL_FLOAT_TYPES(construct);
   TEST_ALL_FLOAT_TYPES(assign);
   TEST_ALL_TYPES(resize);
   TEST_ALL_TYPES(common);
   TEST_ALL_TYPES(index);
   TEST(valid);
   TEST(same);
   TEST(swap);
   TEST(copy);
   TEST_ALL_FLOAT_TYPES(slice);

   return EXIT_SUCCESS;
}
