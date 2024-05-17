#pragma once

#include "../src/tensor.cuh"

using namespace tnb;


void same_tensor_extents() {
   Tensor<float, 3> A(3, 3, 3), B(3, 3, 3);
   Tensor<double, 3> C(3, 3, 3), D(3, 3, 4);
   assert(same_extents(A, B, C));

   assert(same_extents(A, B, C));
   assert(!same_extents(A, B, D));
}


void same_tensor_value_type() {
   Tensor<float, 3> x, y, z;
   Tensor<double, 3> w;
   static_assert(same_value_type<decltype(x), decltype(y), decltype(z)>());
   static_assert(!same_value_type<decltype(x), decltype(y), decltype(z), decltype(w)>());
}


void same_tensor_memory_kind() {
   Tensor<float, 3> x, y, z;
   CudaTensor<double, 3> w;
   static_assert(same_memory_kind<decltype(x), decltype(y), decltype(z)>());
   static_assert(!same_memory_kind<decltype(x), decltype(y), decltype(z), decltype(w)>());
}


void same() {
   same_tensor_extents();
   same_tensor_value_type();
   same_tensor_memory_kind();
}

