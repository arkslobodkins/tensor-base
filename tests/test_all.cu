#include <cstdlib>

#include "assign.cuh"
#include "common.cuh"
#include "construct.cuh"
#include "index.cuh"
#include "resize.cuh"
#include "test.cuh"

int main() {
   TEST_ALL_FLOAT_TYPES(construct);
   TEST_ALL_FLOAT_TYPES(assign);
   TEST_ALL_TYPES(resize);
   TEST_ALL_TYPES(common);
   TEST_ALL_TYPES(index);
   return EXIT_SUCCESS;
}
