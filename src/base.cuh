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
__host__ __device__ constexpr index_t sizeof_cast() {
   return static_cast<index_t>(sizeof...(Types));
}


template <typename IntType>
__host__ __device__ constexpr index_t index_cast(IntType i) {
   return static_cast<index_t>(i);
}


template <typename T>
__host__ __device__ constexpr bool is_compatible_integer() {
   return std::is_same_v<short int, T> || std::is_same_v<unsigned short int, T> || std::is_same_v<int, T>
       || std::is_same_v<unsigned int, T> || std::is_same_v<long int, T>;
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
      static_assert((... && is_compatible_integer<Ints>()));
      static_assert(sizeof_cast<Ints...>() == dim);
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

   __host__ __device__ bool operator==(const Extents& ext) {
      for(index_t i = 0; i < dim; ++i) {
         if((*this)[i] != ext[i]) {
            return false;
         }
      }
      return true;
   }

   __host__ __device__ bool operator!=(const Extents& ext) {
      return !(*this == ext);
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


template <typename T, typename... Ts>
__host__ __device__ constexpr bool same_value_type() {
   return (std::is_same_v<ValueTypeOf<T>, ValueTypeOf<Ts>> && ...);
}


template <typename T, typename... Ts>
__host__ __device__ constexpr bool same_memory_kind() {
   return ((T::memory_kind() == Ts::memory_kind()) && ...);
}


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
      return this->size() == 0 ? nullptr : data_;
   }


   __host__ __device__ const_ptr_t data() const {
      return this->size() == 0 ? nullptr : data_;
   }


   __host__ __device__ bool empty() const {
      return !this->size();
   }


   __host__ __device__ cnd_ptr_t begin() {
      return this->data();
   }


   __host__ __device__ const_ptr_t begin() const {
      return this->data();
   }


   __host__ __device__ const_ptr_t cbegin() const {
      return this->data();
   }


   //  if size() is 0 then data() is nullptr, thus nullptr + 0 is safe
   __host__ __device__ cnd_ptr_t end() {
      return this->data() + this->size();
   }


   __host__ __device__ const_ptr_t end() const {
      return this->data() + this->size();
   }


   __host__ __device__ const_ptr_t cend() const {
      return this->data() + this->size();
   }


   ////////////////////////////////////////////////////////////////////////////////////////////////
   __host__ __device__ static constexpr bool is_host() {
      return scheme == host;
   }


   __host__ __device__ static constexpr bool is_device() {
      return scheme == device;
   }


   __host__ __device__ static constexpr bool is_unified() {
      return scheme == unified;
   }


   __host__ __device__ static constexpr auto memory_kind() {
      return scheme;
   }


   __host__ __device__ static constexpr index_t dimension() {
      return dim;
   }


   __host__ __device__ index_t size() const {
      return ext_.size();
   }


   __host__ __device__ index_t bytes() const {
      return this->size() * sizeof(T);
   }


   __host__ __device__ index_t extent(index_t d) const {
      return ext_[d];
   }


   __host__ __device__ Extents<dim> extents() const {
      return ext_;
   }


   ////////////////////////////////////////////////////////////////////////////////////////////////
   __host__ __device__ bool valid_index(index_t i, index_t d) const {
      assert(d > -1 && d < dim);
      return i > -1 && i < ext_[d];
   }


   template <index_t in_dim, typename First, typename... Ints>
   __host__ __device__ index_t offset_of(First first, Ints... indexes) const {
      static_assert(is_compatible_integer<First>());
      assert(this->valid_index(first, in_dim - 1 - sizeof_cast<Ints...>()));

      if constexpr(sizeof...(Ints) == 0) {
         return first * ext_.product_from(in_dim);
      } else {
         return first * ext_.product_from(in_dim - sizeof_cast<Ints...>())
              + this->offset_of<in_dim, Ints...>(indexes...);
      }
   }


   template <typename First, typename... Ints>
   __host__ __device__ index_t index_of(First first, Ints... indexes) const {
      return this->offset_of<dim, First, Ints...>(first, indexes...);
   }


   template <typename... Ints>
   __host__ __device__ cnd_ref_t operator()(Ints... indexes) {
      TENSOR_STATIC_ASSERT_DIMENSION();
      return data_[this->index_of(indexes...)];
   }


   template <typename... Ints>
   __host__ __device__ const_ref_t operator()(Ints... indexes) const {
      TENSOR_STATIC_ASSERT_DIMENSION();
      return data_[this->index_of(indexes...)];
   }


   template <typename Int>
   __host__ __device__ cnd_ref_t operator[](Int i) {
      static_assert(is_compatible_integer<Int>());
      assert(index_cast(i) > -1 && index_cast(i) < this->size());
      return data_[i];
   }


   template <typename Int>
   __host__ __device__ const_ref_t operator[](Int i) const {
      static_assert(is_compatible_integer<Int>());
      assert(index_cast(i) > -1 && index_cast(i) < this->size());
      return data_[i];
   }


   __host__ void memset_sync(int val) {
      if constexpr(this->is_host()) {
         std::memset(this->data(), val, this->size() * sizeof(T));
      } else {
         // set on device for unified memory type
         ASSERT_CUDA(cudaMemset(this->data(), val, this->size() * sizeof(T)));
      }
   }


   template <typename TT>
   __host__ void copy_sync(const TT& A) {
      static_assert(std::is_same_v<value_type, ValueTypeOf<TT>>);
      assert(same_extents(*this, A));

      if constexpr(this->is_device() && A.is_device()) {
         ASSERT_CUDA(cudaMemcpy(this->data(), A.data(), this->bytes(), cudaMemcpyDeviceToDevice));

      } else if constexpr(this->is_host() && A.is_device()) {
         ASSERT_CUDA(cudaMemcpy(this->data(), A.data(), this->bytes(), cudaMemcpyDeviceToHost));

      } else if constexpr(this->is_device() && A.is_host()) {
         ASSERT_CUDA(cudaMemcpy(this->data(), A.data(), this->bytes(), cudaMemcpyHostToDevice));

      } else if constexpr(this->is_host() && A.is_host()) {
         ASSERT_CUDA(cudaMemcpy(this->data(), A.data(), this->bytes(), cudaMemcpyHostToHost));
      } else if constexpr(this->is_unified() && A.is_unified()) {
         ASSERT_CUDA(cudaMemcpy(this->data(), A.data(), this->bytes(), cudaMemcpyDeviceToDevice));
      } else {
         // copy_sync is allowed for unified memory types only if both types are unified
         static_assert_false<T>();
      }
   }
};


template <typename T, index_t dim, Scheme scheme, bool is_const_ptr = false, bool is_pinned_v = false>
class LinearBase : public LinearBaseCommon<T, dim, scheme, is_const_ptr> {
   static_assert(scheme == host || scheme == device);
   static_assert(is_pinned_v == true ? (scheme == host) : true);

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


   __host__ __device__ index_t bytes() const {
      TENSOR_VALIDATE_HOST_DEBUG;
      return Base::bytes();
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


   static constexpr __host__ __device__ bool is_pinned() {
      static_assert(Base::is_host());
      return is_pinned_v;
   }


   template <typename TT>
   __host__ void copy_async(const TT& A, cudaStream_t stream = 0) {
      static_assert(!TT::is_unified());
      static_assert(std::is_same_v<value_type, ValueTypeOf<TT>>);
      assert(same_extents(*this, A));

      if constexpr(this->is_device() && A.is_device()) {
         ASSERT_CUDA(
             cudaMemcpyAsync(this->data(), A.data(), this->bytes(), cudaMemcpyDeviceToDevice, stream));

      } else if constexpr(this->is_host() && A.is_device()) {
         static_assert(this->is_pinned());
         ASSERT_CUDA(cudaMemcpyAsync(this->data(), A.data(), this->bytes(), cudaMemcpyDeviceToHost, stream));

      } else if constexpr(this->is_device() && A.is_host()) {
         static_assert(A.is_pinned());
         ASSERT_CUDA(cudaMemcpyAsync(this->data(), A.data(), this->bytes(), cudaMemcpyHostToDevice, stream));

      } else {
         // host to host copy is not asynchronous
         static_assert_false<T>();
      }
   }


   __host__ void memset_async(int val, cudaStream_t stream = 0) {
      static_assert(this->is_device());
      ASSERT_CUDA(cudaMemsetAsync(this->data(), val, this->size() * sizeof(T), stream));
   }
};


}  // namespace tnb
