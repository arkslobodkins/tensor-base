// Arkadijs Slobodkins
// 2024

#pragma once


#include <cassert>
#include <cstring>
#include <type_traits>

#include "common.cuh"
#include "extents.cuh"


#define TENSOR_STATIC_ASSERT_DIMENSION()                       \
   static_assert(static_cast<index_t>(sizeof...(Ints)) == dim, \
                 "index dimension must be equal to the dimension of the tensor")


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


template <typename T>
using ValueTypeOf = typename T::value_type;


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


template <typename T, typename... Ts>
__host__ __device__ constexpr bool same_value_type() {
   return (std::is_same_v<ValueTypeOf<T>, ValueTypeOf<Ts>> && ...);
}


template <typename T, typename... Ts>
__host__ __device__ constexpr bool same_memory_kind() {
   return ((T::memory_kind() == Ts::memory_kind()) && ...);
}


enum Scheme { Host, Device, Unified };


template <typename T, index_t dim, Scheme scheme, bool is_const_ptr = false,
          bool is_pinned_mem = false>
class LinearBaseCommon {
private:
   using Self = LinearBaseCommon<T, dim, scheme, is_const_ptr, is_pinned_mem>;

   using cnd_ptr_t = std::conditional_t<is_const_ptr, const T*, T*>;
   using cnd_ref_t = std::conditional_t<is_const_ptr, const T&, T&>;

   using const_ptr_t = const T*;
   using const_ref_t = const T&;

protected:
   Extents<dim> ext_{};
   cnd_ptr_t data_{};

   __host__ __device__ static void validate_host_type() {
#ifdef __CUDA_ARCH__
      if constexpr(scheme == Host) {
         __device__ void not_callable_on_device_error();
         not_callable_on_device_error();
      }
#endif
   }

   __host__ __device__ static void validate_device_type() {
#ifndef __CUDA_ARCH__
      if constexpr(scheme != Host) {
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
   static_assert(is_pinned_mem == true ? (scheme == Host) : true);

   using size_type = index_t;
   using value_type = T;


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


   //  If size() is 0 then data() is nullptr, thus nullptr + 0 is safe.
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
      return scheme == Host;
   }


   __host__ __device__ static constexpr bool is_device() {
      return scheme == Device;
   }


   __host__ __device__ static constexpr bool is_unified() {
      return scheme == Unified;
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
      static_assert(internal::is_compatible_integer<First>());
      assert(this->valid_index(first, in_dim - 1 - internal::sizeof_cast<Ints...>()));

      if constexpr(sizeof...(Ints) == 0) {
         return first * ext_.product_from(in_dim);
      } else {
         return first * ext_.product_from(in_dim - internal::sizeof_cast<Ints...>())
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
      static_assert(internal::is_compatible_integer<Int>());
      assert(internal::index_cast(i) > -1 && internal::index_cast(i) < this->size());
      return data_[i];
   }


   template <typename Int>
   __host__ __device__ const_ref_t operator[](Int i) const {
      static_assert(internal::is_compatible_integer<Int>());
      assert(internal::index_cast(i) > -1 && internal::index_cast(i) < this->size());
      return data_[i];
   }


   __host__ void memset(int val) {
      if constexpr(this->is_host()) {
         std::memset(this->data(), val, this->size() * sizeof(T));
      } else {
         // Set on device for device or unified memory type.
         ASSERT_CUDA(cudaMemset(this->data(), val, this->size() * sizeof(T)));
      }
   }


   __host__ void memset_async(int val, cudaStream_t stream = 0) {
      static_assert(!this->is_host());
      ASSERT_CUDA(cudaMemsetAsync(this->data(), val, this->size() * sizeof(T), stream));
   }


   template <typename TT>
   __host__ void copy_sync(const TT& A) {
      static_assert(std::is_same_v<value_type, ValueTypeOf<TT>>);
      assert(same_extents(*this, A));
      ASSERT_CUDA(cudaMemcpy(this->data(), A.data(), this->bytes(), cudaMemcpyDefault));
   }


   template <typename TT>
   __host__ void copy_async(const TT& A, cudaStream_t stream = 0) {
      // Host to host copy is not asynchronous.
      static_assert(!(this->is_host() && TT::is_host());
      static_assert(std::is_same_v<value_type, ValueTypeOf<TT>>);
      assert(same_extents(*this, A));

      if constexpr(this->is_host()) {
         static_assert(this->is_pinned());
      } else if constexpr(TT::is_host()) {
         static_assert(TT::is_pinned());
      }
      ASSERT_CUDA(
          cudaMemcpyAsync(this->data(), A.data(), this->bytes(), cudaMemcpyDefault, stream));
   }


   static constexpr __host__ __device__ bool is_pinned() {
      static_assert(Self::is_host());
      return is_pinned_mem;
   }
};


template <typename T, index_t dim, Scheme scheme, bool is_const_ptr = false,
          bool is_pinned_mem = false>
class LinearBase : public LinearBaseCommon<T, dim, scheme, is_const_ptr, is_pinned_mem> {
private:
   using Base = LinearBaseCommon<T, dim, scheme, is_const_ptr, is_pinned_mem>;

protected:
   using Base::data_;
   using Base::ext_;

public:
   static_assert(scheme == Host || scheme == Device);
   static_assert(is_pinned_mem == true ? (scheme == Host) : true);

   using typename Base::size_type;
   using typename Base::value_type;


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
};


}  // namespace tnb
