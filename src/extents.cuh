// Arkadijs Slobodkins
// 2024

#pragma once


#include <cassert>

#include "common.cuh"


namespace tnb {


template <index_t dim>
class Extents {
public:
   static_assert(dim > 0);

   Extents() = default;
   Extents(const Extents&) = default;
   Extents& operator=(const Extents&) = default;

   template <typename... Ints>
   __host__ __device__ constexpr explicit Extents(Ints... ext) : x_{ext...} {
      static_assert((... && internal::is_compatible_integer<Ints>()));
      static_assert(internal::sizeof_cast<Ints...>() == dim);
   }

   __host__ __device__ index_t& operator[](index_t d) {
      assert(d > -1 && d < dim);
      return x_[d];
   }

   __host__ __device__ const index_t& operator[](index_t d) const {
      assert(d > -1 && d < dim);
      return x_[d];
   }

   __host__ __device__ index_t product_from(index_t n) const {
      assert(n > -1 && n <= dim);  // n == dim is allowed
      index_t p = 1;
      for(index_t d = n; d < dim; ++d) {
         p *= x_[d];
      }
      return p;
   }

   __host__ __device__ index_t size() const {
      return this->product_from(0);
   }

   __host__ __device__ friend bool operator==(const Extents& ext1, const Extents& ext2) {
      for(index_t i = 0; i < dim; ++i) {
         if(ext1[i] != ext2[i]) {
            return false;
         }
      }
      return true;
   }

   __host__ __device__ friend bool operator!=(const Extents& ext1, const Extents& ext2) {
      return !(ext1 == ext2);
   }

private:
   index_t x_[dim]{};
};


template <typename... Ints>
__host__ __device__ bool valid_extents(Ints... ext) {
   // Non-negative and either all nonzero or zero.
   return (... && (ext > -1)) && ((... && (ext != 0)) || (... && (ext == 0)));
}


template <index_t dim>
__host__ __device__ bool valid_extents(const Extents<dim>& ext) {
   for(index_t d = 0; d < dim; ++d) {
      if(ext[d] < 0) {
         return false;
      }
   }

   bool cnd = (ext[0] == 0);
   for(index_t d = 1; d < dim; ++d) {
      if((ext[d] == 0) ^ cnd) {
         return false;
      }
   }
   return true;
}


}  // namespace tnb
