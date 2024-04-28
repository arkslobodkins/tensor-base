// Arkadijs Slobodkins
// 2024

#pragma once


#include <cuda_runtime.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <type_traits>


#define ASSERT_CUDA_SUCCESS(cudaCall)                                                                   \
   do {                                                                                                 \
      cudaError_t error = cudaCall;                                                                     \
      if(error != cudaSuccess) {                                                                        \
         std::fprintf(                                                                                  \
             stderr, "Error on line %i, file %s: %s\n", __LINE__, __FILE__, cudaGetErrorString(error)); \
         std::exit(EXIT_FAILURE);                                                                       \
      }                                                                                                 \
   } while(0)


#define TENSOR_STATIC_ASSERT_DIMENSION() \
   static_assert(sizeof...(Ints) == dim, "index dimension must be equal to the dimension of the tensor")


namespace tnb {


using index_t = long int;


template <typename T>
__host__ __device__ constexpr bool is_actually_integer() {
   // not checking for char8_t, which is only available since C++20
   return std::numeric_limits<T>::is_integer && (!std::is_same_v<T, bool>) && (!std::is_same_v<T, char>)
       && (!std::is_same_v<T, signed char>) && (!std::is_same_v<T, unsigned char>)
       && (!std::is_same_v<T, wchar_t>) && (!std::is_same_v<T, char16_t>) && (!std::is_same_v<T, char32_t>);
}


template <index_t dim>
struct Extents {
private:
   index_t x[dim]{};

public:
   template <typename... Ints>
   __host__ __device__ Extents(Ints... ext) : x{ext...} {
      static_assert((... && is_actually_integer<Ints>()));
      static_assert(static_cast<index_t>(sizeof...(ext)) == dim);
      assert((... && (ext != 0)) || (... && (ext == 0)));  // either all zero or nonzero
   }

   explicit Extents() = default;
   Extents(const Extents&) = default;
   Extents& operator=(const Extents&) = default;

   __host__ __device__ const index_t& operator[](index_t d) const {
      assert(d > -1 && d < dim);
      return x[d];
   }

   __host__ __device__ index_t& operator[](index_t d) {
      assert(d > -1 && d < dim);
      return x[d];
   }

   __host__ __device__ index_t product_from(index_t n) const {
      assert(n > -1 && n <= dim);  // n = dim is allowed
      index_t p = 1;
      for(index_t d = n; d < dim; ++d) {
         p *= x[d];
      }
      return p;
   }
};


template <typename TensorType1, typename TensorType2>
__host__ __device__ bool same_extents(const TensorType1& A, const TensorType2& B) {
   if(A.dimension() != B.dimension()) {
      return false;
   }
   for(index_t d = 0; d < A.dimension(); ++d) {
      if(A.extent(d) != B.extent(d)) {
         return false;
      }
   }
   return true;
}


template <typename T, index_t dim, bool is_host_t, bool is_const_ptr = false>
class LinearBase {
private:
   using cnd_ptr_t = std::conditional_t<is_const_ptr, const T*, T*>;
   using cnd_ref_t = std::conditional_t<is_const_ptr, const T&, T&>;

   using const_ptr_t = const T*;
   using const_ref_t = const T&;

protected:
   Extents<dim> ext_{};
   cnd_ptr_t data_{};

   static void __host__ __device__ validate_host_type() {
#ifdef __CUDA_ARCH__
      if constexpr(is_host_t) {
         __device__ void not_callable_on_device_error();
         not_callable_on_device_error();
      }
#endif
   }

   static void __host__ __device__ validate_device_type() {
#ifndef __CUDA_ARCH__
      if constexpr(!is_host_t) {
         __host__ void not_callable_on_host_error();
         not_callable_on_host_error();
      }
#endif
   }

   static void __host__ __device__ validate_host_device_type() {
      validate_host_type();
      validate_device_type();
   }

public:
   using size_type = index_t;
   using value_type = T;


   ////////////////////////////////////////////////////////////////////////////////////////////////
   explicit LinearBase() = default;
   LinearBase(const LinearBase&) = default;
   LinearBase& operator=(const LinearBase&) = default;


   ////////////////////////////////////////////////////////////////////////////////////////////////
   __host__ __device__ cnd_ptr_t data() {
      validate_host_type();
      return data_;
   }


   __host__ __device__ const_ptr_t data() const {
      validate_host_type();
      return data_;
   }


   __host__ __device__ bool empty() const {
      validate_host_type();
      return size();
   }


   __host__ __device__ cnd_ptr_t begin() {
      validate_host_device_type();
      return data_;
   }


   __host__ __device__ const_ptr_t begin() const {
      validate_host_device_type();
      return data_;
   }


   __host__ __device__ const_ptr_t cbegin() const {
      return begin();
   }


   __host__ __device__ cnd_ptr_t end() {
      validate_host_device_type();
      return data_ + size();
   }


   __host__ __device__ const_ptr_t end() const {
      validate_host_device_type();
      return data_ + size();
   }


   __host__ __device__ const_ptr_t cend() const {
      return end();
   }


   ////////////////////////////////////////////////////////////////////////////////////////////////
   __host__ __device__ static constexpr bool host_type() {
      // validate_host_type();
      return is_host_t;
   }


   __host__ __device__ static constexpr bool device_type() {
      // validate_host_type();
      return !is_host_t;
   }


   __host__ __device__ static constexpr index_t dimension() {
      // validate_host_type();
      return dim;
   }


   __host__ __device__ index_t size() const {
      validate_host_type();
      return ext_.product_from(0);
   }


   __host__ __device__ index_t extent(index_t d) const {
      validate_host_type();
      return ext_[d];
   }


   __host__ __device__ Extents<dim> extents() const {
      validate_host_type();
      return ext_;
   }


   ////////////////////////////////////////////////////////////////////////////////////////////////
   __host__ __device__ bool valid_index(index_t i, index_t d) const {
      validate_host_type();
      return i > -1 && i < ext_[d];
   }


   template <index_t in_dim, typename First, typename... Ints>
   __host__ __device__ index_t offset_of(First first, Ints... indexes) const {
      validate_host_type();
      static_assert(is_actually_integer<First>());
      assert(valid_index(first, in_dim - 1 - static_cast<index_t>(sizeof...(Ints))));

      if constexpr(sizeof...(Ints) == 0) {
         return first * ext_.product_from(in_dim);
      } else {
         return first * ext_.product_from(in_dim - static_cast<index_t>(sizeof...(Ints)))
              + offset_of<in_dim, Ints...>(indexes...);
      }
   }


   template <typename First, typename... Ints>
   __host__ __device__ index_t index_of(First first, Ints... indexes) const {
      validate_host_type();
      return offset_of<dim, First, Ints...>(first, indexes...);
   }


   template <typename... Ints>
   __host__ __device__ cnd_ref_t operator()(Ints... indexes) {
      validate_host_device_type();
      TENSOR_STATIC_ASSERT_DIMENSION();
      return data_[index_of(indexes...)];
   }


   template <typename... Ints>
   __host__ __device__ const_ref_t operator()(Ints... indexes) const {
      validate_host_device_type();
      TENSOR_STATIC_ASSERT_DIMENSION();
      return data_[index_of(indexes...)];
   }


   template <typename TensorType>
   __host__ void copy_sync(const TensorType& A) {
      static_assert(dimension() == A.dimension());
      assert(same_extents(*this, A));

      auto nbytes = size() * sizeof(T);
      if constexpr(device_type() && A.device_type()) {
         ASSERT_CUDA_SUCCESS(cudaMemcpy(data(), A.data(), nbytes, cudaMemcpyDeviceToDevice));

      } else if constexpr(host_type() && A.device_type()) {
         ASSERT_CUDA_SUCCESS(cudaMemcpy(data(), A.data(), nbytes, cudaMemcpyDeviceToHost));

      } else if constexpr(device_type() && A.host_type()) {
         ASSERT_CUDA_SUCCESS(cudaMemcpy(data(), A.data(), nbytes, cudaMemcpyHostToDevice));

      } else {
         ASSERT_CUDA_SUCCESS(cudaMemcpy(data(), A.data(), nbytes, cudaMemcpyHostToHost));
      }
   }
};


}  // namespace tnb
