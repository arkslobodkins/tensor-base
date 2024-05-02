// Arkadijs Slobodkins
// 2024

#pragma once


#include <cuda_runtime.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>


#define ASSERT_CUDA(cudaCall)                                                                   \
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


#ifndef NDEBUG
#define TENSOR_VALIDATE_HOST_DEBUG this->validate_host_type()
#else
#define TENSOR_VALIDATE_HOST_DEBUG
#endif


#ifndef NDEBUG
#define TENSOR_VALIDATE_HOST_DEVICE_DEBUG validate_host_device_type()
#else
#define TENSOR_VALIDATE_HOST_DEVICE_DEBUG
#endif


namespace tnb {


using index_t = long int;


template <typename... Types>
__host__ __device__ constexpr index_t SizeOfCast() {
   return static_cast<index_t>(sizeof...(Types));
}


template <typename T>
__host__ __device__ constexpr bool is_actually_integer() {
   // not checking for char8_t, which is only available since C++20
   return std::numeric_limits<T>::is_integer && (!std::is_same_v<T, bool>) && (!std::is_same_v<T, char>)
       && (!std::is_same_v<T, signed char>) && (!std::is_same_v<T, unsigned char>)
       && (!std::is_same_v<T, wchar_t>) && (!std::is_same_v<T, char16_t>) && (!std::is_same_v<T, char32_t>);
}


template <index_t dim>
struct Extents {
   static_assert(dim > 0);

private:
   index_t x_[dim]{};

public:
   template <typename... Ints>
   __host__ __device__ Extents(Ints... ext) : x_{ext...} {
      static_assert((... && is_actually_integer<Ints>()));
      static_assert(SizeOfCast<Ints...>() == dim);
   }

   explicit Extents() = default;
   Extents(const Extents&) = default;
   Extents& operator=(const Extents&) = default;

   __host__ __device__ index_t& operator[](index_t d) {
      assert(d > -1 && d < dim);
      return x_[d];
   }

   __host__ __device__ const index_t& operator[](index_t d) const {
      assert(d > -1 && d < dim);
      return x_[d];
   }

   __host__ __device__ index_t product_from(index_t n) const {
      assert(n > -1 && n <= dim);  // n = dim is allowed
      index_t p = 1;
      for(index_t d = n; d < dim; ++d) {
         p *= x_[d];
      }
      return p;
   }

   __host__ __device__ index_t size() const {
      return product_from(0);
   }
};


template <typename First, typename Second, typename... TensorTypes>
__host__ __device__ bool same_extents(const First& first, const Second& second,
                                      const TensorTypes&... tensors) {
   if(first.dimension() != second.dimension()) {
      return false;
   }
   for(index_t d = 0; d < first.dimension(); ++d) {
      if(first.extent(d) != second.extent(d)) {
         return false;
      }
   }
   if constexpr(sizeof...(TensorTypes) == 0) {
      return true;
   } else {
      return same_extents(second, tensors...);
   }
}


template <typename... Ints>
__host__ __device__ bool valid_extents(Ints... ext) {
   // non-negative and either all zero or nonzero
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

   __host__ __device__ static void validate_host_type() {
#ifdef __CUDA_ARCH__
      if constexpr(is_host_t) {
         __device__ void not_callable_on_device_error();
         not_callable_on_device_error();
      }
#endif
   }

   __host__ __device__ static void validate_device_type() {
#ifndef __CUDA_ARCH__
      if constexpr(!is_host_t) {
         __host__ void not_callable_on_host_error();
         not_callable_on_host_error();
      }
#endif
   }

   __host__ __device__ static void validate_host_device_type() {
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
      TENSOR_VALIDATE_HOST_DEBUG;
      return size() == 0 ? nullptr : data_;
   }


   __host__ __device__ const_ptr_t data() const {
      TENSOR_VALIDATE_HOST_DEBUG;
      return size() == 0 ? nullptr : data_;
   }


   __host__ __device__ bool empty() const {
      TENSOR_VALIDATE_HOST_DEBUG;
      return !size();
   }


   __host__ __device__ cnd_ptr_t begin() {
      TENSOR_VALIDATE_HOST_DEVICE_DEBUG;
      return data();
   }


   __host__ __device__ const_ptr_t begin() const {
      TENSOR_VALIDATE_HOST_DEVICE_DEBUG;
      return data();
   }


   __host__ __device__ const_ptr_t cbegin() const {
      return begin();
   }


   __host__ __device__ cnd_ptr_t end() {
      TENSOR_VALIDATE_HOST_DEVICE_DEBUG;
      return data() + size();
   }


   __host__ __device__ const_ptr_t end() const {
      TENSOR_VALIDATE_HOST_DEVICE_DEBUG;
      return data() + size();
   }


   __host__ __device__ const_ptr_t cend() const {
      return end();
   }


   ////////////////////////////////////////////////////////////////////////////////////////////////
   __host__ __device__ static constexpr bool host_type() {
      return is_host_t;
   }


   __host__ __device__ static constexpr bool device_type() {
      return !is_host_t;
   }


   __host__ __device__ static constexpr index_t dimension() {
      return dim;
   }


   __host__ __device__ index_t size() const {
      TENSOR_VALIDATE_HOST_DEBUG;
      return ext_.size();
   }


   __host__ __device__ index_t extent(index_t d) const {
      TENSOR_VALIDATE_HOST_DEBUG;
      return ext_[d];
   }


   __host__ __device__ Extents<dim> extents() const {
      TENSOR_VALIDATE_HOST_DEBUG;
      return ext_;
   }


   ////////////////////////////////////////////////////////////////////////////////////////////////
   __host__ __device__ bool valid_index(index_t i, index_t d) const {
      TENSOR_VALIDATE_HOST_DEBUG;
      return i > -1 && i < ext_[d];
   }


   template <index_t in_dim, typename First, typename... Ints>
   __host__ __device__ index_t offset_of(First first, Ints... indexes) const {
      TENSOR_VALIDATE_HOST_DEBUG;
      static_assert(is_actually_integer<First>());
      assert(valid_index(first, in_dim - 1 - SizeOfCast<Ints...>()));

      if constexpr(sizeof...(Ints) == 0) {
         return first * ext_.product_from(in_dim);
      } else {
         return first * ext_.product_from(in_dim - SizeOfCast<Ints...>())
              + offset_of<in_dim, Ints...>(indexes...);
      }
   }


   template <typename First, typename... Ints>
   __host__ __device__ index_t index_of(First first, Ints... indexes) const {
      TENSOR_VALIDATE_HOST_DEBUG;
      return offset_of<dim, First, Ints...>(first, indexes...);
   }


   template <typename... Ints>
   __host__ __device__ cnd_ref_t operator()(Ints... indexes) {
      TENSOR_VALIDATE_HOST_DEVICE_DEBUG;
      TENSOR_STATIC_ASSERT_DIMENSION();
      return data_[index_of(indexes...)];
   }


   template <typename... Ints>
   __host__ __device__ const_ref_t operator()(Ints... indexes) const {
      TENSOR_VALIDATE_HOST_DEVICE_DEBUG;
      TENSOR_STATIC_ASSERT_DIMENSION();
      return data_[index_of(indexes...)];
   }


   template <typename Int>
   __host__ __device__ cnd_ref_t operator[](Int i) {
      TENSOR_VALIDATE_HOST_DEVICE_DEBUG;
      static_assert(is_actually_integer<Int>());
      assert(i > -1 && i < size());
      return data_[i];
   }


   template <typename Int>
   __host__ __device__ const_ref_t operator[](Int i) const {
      TENSOR_VALIDATE_HOST_DEVICE_DEBUG;
      static_assert(is_actually_integer<Int>());
      assert(i > -1 && i < size());
      return data_[i];
   }


   template <template <typename, index_t> class TensorType>
   __host__ void copy_sync(const TensorType<T, dim>& A) {
      assert(same_extents(*this, A));

      auto nbytes = size() * sizeof(T);
      if constexpr(device_type() && A.device_type()) {
         ASSERT_CUDA(cudaMemcpy(data(), A.data(), nbytes, cudaMemcpyDeviceToDevice));

      } else if constexpr(host_type() && A.device_type()) {
         ASSERT_CUDA(cudaMemcpy(data(), A.data(), nbytes, cudaMemcpyDeviceToHost));

      } else if constexpr(device_type() && A.host_type()) {
         ASSERT_CUDA(cudaMemcpy(data(), A.data(), nbytes, cudaMemcpyHostToDevice));

      } else {
         ASSERT_CUDA(cudaMemcpy(data(), A.data(), nbytes, cudaMemcpyHostToHost));
      }
   }


   __host__ void memset_sync(int val) {
      if constexpr(device_type()) {
         ASSERT_CUDA(cudaMemset(data(), val, size() * sizeof(T)));
      } else {
         std::memset(data(), val, size() * sizeof(T));
      }
   }
};


}  // namespace tnb
