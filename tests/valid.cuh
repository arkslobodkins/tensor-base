#pragma once


#include "../src/tnb.cuh"


using namespace tnb;


void valid_tensor_extents() {
   assert(valid_extents(Extents<4>{1, 1, 1, 1}));
   assert(valid_extents(Extents<4>{0, 0, 0, 0}));
   assert(!valid_extents(Extents<4>{1, 1, 1, 0}));

   assert(valid_extents(1, 1, 1, 1));
   assert(valid_extents(0, 0, 0, 0));
   assert(!valid_extents(1, 1, 1, 0));
}


void valid_tensor_index() {
   Tensor<int, 2> x(4, 4);
   assert(x.valid_index(0, 0));
   assert(x.valid_index(0, 1));
   assert(x.valid_index(3, 0));
   assert(x.valid_index(3, 1));

   assert(!x.valid_index(-1, 0));
   assert(!x.valid_index(-1, 1));
   assert(!x.valid_index(4, 0));
   assert(!x.valid_index(4, 1));
}


void valid() {
   valid_tensor_extents();
   valid_tensor_index();
}

