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


#define ASSERT_CUDA(cudaCall)                                                                           \
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
#define TENSOR_VALIDATE_HOST_DEVICE_DEBUG this->validate_host_device_type()
#else
#define TENSOR_VALIDATE_HOST_DEVICE_DEBUG
#endif


namespace tnb {


using index_t = long int;


template <typename T>
using ValueTypeOf = typename T::value_type;


template <typename... Types>
__host__ __device__ constexpr index_t SizeOfCast() {
   return static_cast<index_t>(sizeof...(Types));
}


template <typename IntType>
__host__ __device__ constexpr index_t index_cast(IntType i) {
   return static_cast<index_t>(i);
}


template <typename T>
__host__ __device__ constexpr bool is_actually_integer() {
   // not checking for char8_t, which is only available since C++20
   return std::numeric_limits<T>::is_integer && (!std::is_same_v<T, bool>) && (!std::is_same_v<T, char>)
       && (!std::is_same_v<T, signed char>) && (!std::is_same_v<T, unsigned char>)
       && (!std::is_same_v<T, wchar_t>) && (!std::is_same_v<T, char16_t>) && (!std::is_same_v<T, char32_t>);
}


template <typename T>
__host__ __device__ constexpr void static_assert_false() {
   static_assert(!sizeof(T));
}


template <index_t dim>
struct Extents {
   static_assert(dim > 0);

private:
   index_t x_[dim]{};

public:
   explicit Extents() = default;
   Extents(const Extents&) = default;
   Extents& operator=(const Extents&) = default;

   template <typename... Ints>
   __host__ __device__ constexpr Extents(Ints... ext) : x_{ext...} {
      static_assert((... && is_actually_integer<Ints>()));
      static_assert(SizeOfCast<Ints...>() == dim);
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


enum Scheme { host, device, unified };


template <typename T, index_t dim, Scheme scheme, bool is_const_ptr = false>
class LinearBaseCommon {
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
      if constexpr(scheme == host) {
         __device__ void not_callable_on_device_error();
         not_callable_on_device_error();
      }
#endif
   }

   __host__ __device__ static void validate_device_type() {
#ifndef __CUDA_ARCH__
      if constexpr(!scheme == host) {
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
   explicit LinearBaseCommon() = default;
   LinearBaseCommon(const LinearBaseCommon&) = default;
   LinearBaseCommon& operator=(const LinearBaseCommon&) = default;


   ////////////////////////////////////////////////////////////////////////////////////////////////
   __host__ __device__ cnd_ptr_t data() {
      return size() == 0 ? nullptr : data_;
   }


   __host__ __device__ const_ptr_t data() const {
      return size() == 0 ? nullptr : data_;
   }


   __host__ __device__ bool empty() const {
      return !size();
   }


   __host__ __device__ cnd_ptr_t begin() {
      return data();
   }


   __host__ __device__ const_ptr_t begin() const {
      return data();
   }


   __host__ __device__ const_ptr_t cbegin() const {
      return data();
   }


   __host__ __device__ cnd_ptr_t end() {
      return data() + size();
   }


   __host__ __device__ const_ptr_t end() const {
      return data() + size();
   }


   __host__ __device__ const_ptr_t cend() const {
      return data() + size();
   }


   ////////////////////////////////////////////////////////////////////////////////////////////////
   __host__ __device__ static constexpr bool host_type() {
      return scheme == host;
   }


   __host__ __device__ static constexpr bool device_type() {
      return scheme == device;
   }


   __host__ __device__ static constexpr bool unified_type() {
      return scheme == unified;
   }


   __host__ __device__ static constexpr index_t dimension() {
      return dim;
   }


   __host__ __device__ index_t size() const {
      return ext_.size();
   }


   __host__ __device__ index_t extent(index_t d) const {
      return ext_[d];
   }


   __host__ __device__ Extents<dim> extents() const {
      return ext_;
   }


   ////////////////////////////////////////////////////////////////////////////////////////////////
   __host__ __device__ bool valid_index(index_t i, index_t d) const {
      return i > -1 && i < ext_[d];
   }


   template <index_t in_dim, typename First, typename... Ints>
   __host__ __device__ index_t offset_of(First first, Ints... indexes) const {
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
      return offset_of<dim, First, Ints...>(first, indexes...);
   }


   template <typename... Ints>
   __host__ __device__ cnd_ref_t operator()(Ints... indexes) {
      TENSOR_STATIC_ASSERT_DIMENSION();
      return data_[index_of(indexes...)];
   }


   template <typename... Ints>
   __host__ __device__ const_ref_t operator()(Ints... indexes) const {
      TENSOR_STATIC_ASSERT_DIMENSION();
      return data_[index_of(indexes...)];
   }


   template <typename Int>
   __host__ __device__ cnd_ref_t operator[](Int i) {
      static_assert(is_actually_integer<Int>());
      assert(index_cast(i) > -1 && index_cast(i) < size());
      return data_[i];
   }


   template <typename Int>
   __host__ __device__ const_ref_t operator[](Int i) const {
      static_assert(is_actually_integer<Int>());
      assert(index_cast(i) > -1 && index_cast(i) < size());
      return data_[i];
   }
};


template <typename T, index_t dim, Scheme scheme, bool is_const_ptr = false>
class LinearBase : public LinearBaseCommon<T, dim, scheme, is_const_ptr> {
   static_assert(scheme == host || scheme == device);

private:
   using Base = LinearBaseCommon<T, dim, scheme, is_const_ptr>;

protected:
   using Base::data_;
   using Base::ext_;

public:
   using size_type = index_t;
   using value_type = T;


   ////////////////////////////////////////////////////////////////////////////////////////////////
   explicit LinearBase() = default;
   LinearBase(const LinearBase&) = default;
   LinearBase& operator=(const LinearBase&) = default;


   ////////////////////////////////////////////////////////////////////////////////////////////////
   __host__ __device__ decltype(auto) data() {
      TENSOR_VALIDATE_HOST_DEBUG;
      return Base::data();
   }


   __host__ __device__ decltype(auto) data() const {
      TENSOR_VALIDATE_HOST_DEBUG;
      return Base::data();
   }


   __host__ __device__ bool empty() const {
      TENSOR_VALIDATE_HOST_DEBUG;
      return Base::empty();
   }


   __host__ __device__ decltype(auto) begin() {
      TENSOR_VALIDATE_HOST_DEVICE_DEBUG;
      return Base::begin();
   }


   __host__ __device__ decltype(auto) begin() const {
      TENSOR_VALIDATE_HOST_DEVICE_DEBUG;
      return Base::begin();
   }


   __host__ __device__ decltype(auto) cbegin() const {
      TENSOR_VALIDATE_HOST_DEVICE_DEBUG;
      return Base::cbegin();
   }


   __host__ __device__ decltype(auto) end() {
      TENSOR_VALIDATE_HOST_DEVICE_DEBUG;
      return Base::end();
   }


   __host__ __device__ decltype(auto) end() const {
      TENSOR_VALIDATE_HOST_DEVICE_DEBUG;
      return Base::end();
   }


   __host__ __device__ decltype(auto) cend() const {
      TENSOR_VALIDATE_HOST_DEVICE_DEBUG;
      return Base::cend();
   }


   ////////////////////////////////////////////////////////////////////////////////////////////////
   __host__ __device__ index_t size() const {
      TENSOR_VALIDATE_HOST_DEBUG;
      return Base::size();
   }


   __host__ __device__ index_t extent(index_t d) const {
      TENSOR_VALIDATE_HOST_DEBUG;
      return Base::extent(d);
   }


   __host__ __device__ Extents<dim> extents() const {
      TENSOR_VALIDATE_HOST_DEBUG;
      return Base::extents();
   }


   ////////////////////////////////////////////////////////////////////////////////////////////////
   __host__ __device__ bool valid_index(index_t i, index_t d) const {
      TENSOR_VALIDATE_HOST_DEBUG;
      return Base::valid_index(i, d);
   }


   template <index_t in_dim, typename First, typename... Ints>
   __host__ __device__ index_t offset_of(First first, Ints... indexes) const {
      TENSOR_VALIDATE_HOST_DEBUG;
      return Base::template offset_of<in_dim>(first, indexes...);
   }


   template <typename First, typename... Ints>
   __host__ __device__ index_t index_of(First first, Ints... indexes) const {
      TENSOR_VALIDATE_HOST_DEBUG;
      return Base::index_of(first, indexes...);
   }


   template <typename... Ints>
   __host__ __device__ decltype(auto) operator()(Ints... indexes) {
      TENSOR_VALIDATE_HOST_DEVICE_DEBUG;
      return Base::operator()(indexes...);
   }


   template <typename... Ints>
   __host__ __device__ decltype(auto) operator()(Ints... indexes) const {
      TENSOR_VALIDATE_HOST_DEVICE_DEBUG;
      return Base::operator()(indexes...);
   }


   template <typename Int>
   __host__ __device__ decltype(auto) operator[](Int i) {
      TENSOR_VALIDATE_HOST_DEVICE_DEBUG;
      return Base::operator[](i);
   }


   template <typename Int>
   __host__ __device__ decltype(auto) operator[](Int i) const {
      TENSOR_VALIDATE_HOST_DEVICE_DEBUG;
      return Base::operator[](i);
   }


   template <typename TensorType>
   __host__ void copy_sync(const TensorType& A) {
      static_assert(!TensorType::unified_type());
      static_assert(std::is_same_v<value_type, ValueTypeOf<TensorType>>);
      assert(same_extents(*this, A));

      auto nbytes = size() * sizeof(T);
      if constexpr(this->device_type() && A.device_type()) {
         ASSERT_CUDA(cudaMemcpy(data(), A.data(), nbytes, cudaMemcpyDeviceToDevice));

      } else if constexpr(this->host_type() && A.device_type()) {
         ASSERT_CUDA(cudaMemcpy(data(), A.data(), nbytes, cudaMemcpyDeviceToHost));

      } else if constexpr(this->device_type() && A.host_type()) {
         ASSERT_CUDA(cudaMemcpy(data(), A.data(), nbytes, cudaMemcpyHostToDevice));

      } else {
         ASSERT_CUDA(cudaMemcpy(data(), A.data(), nbytes, cudaMemcpyHostToHost));
      }
   }


   template <typename TensorType>
   __host__ void copy_async(const TensorType& A, cudaStream_t stream = 0) {
      static_assert(!TensorType::unified_type());
      static_assert(std::is_same_v<value_type, ValueTypeOf<TensorType>>);
      assert(same_extents(*this, A));

      auto nbytes = size() * sizeof(T);
      if constexpr(this->device_type() && A.device_type()) {
         ASSERT_CUDA(cudaMemcpyAsync(data(), A.data(), nbytes, cudaMemcpyDeviceToDevice, stream));

      } else if constexpr(this->host_type() && A.device_type()) {
         ASSERT_CUDA(cudaMemcpyAsync(data(), A.data(), nbytes, cudaMemcpyDeviceToHost, stream));

      } else if constexpr(this->device_type() && A.host_type()) {
         ASSERT_CUDA(cudaMemcpyAsync(data(), A.data(), nbytes, cudaMemcpyHostToDevice, stream));

      } else {
         ASSERT_CUDA(cudaMemcpyAsync(data(), A.data(), nbytes, cudaMemcpyHostToHost, stream));
      }
   }


   __host__ void memset_sync(int val) {
      if constexpr(this->device_type()) {
         ASSERT_CUDA(cudaMemset(data(), val, size() * sizeof(T)));
      } else {
         std::memset(data(), val, size() * sizeof(T));
      }
   }


   __host__ void memset_async(int val, cudaStream_t stream = 0) {
      static_assert(this->device_type() == true);
      ASSERT_CUDA(cudaMemsetAsync(data(), val, size() * sizeof(T), stream));
   }
};


}  // namespace tnb
