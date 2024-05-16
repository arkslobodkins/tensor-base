#include <cstdlib>

#include "assign.cuh"
#include "construct.cuh"
#include "test.cuh"

int main() {
   TEST_ALL_FLOAT_TYPES(construct);
   TEST_ALL_FLOAT_TYPES(assign);
   return EXIT_SUCCESS;
}
