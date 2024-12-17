// Arkadijs Slobodkins
// 2024


#pragma once


#include <cassert>
#include <cstring>
#include <type_traits>

#include "common.cuh"
#include "extents.cuh"


namespace tnb {


template <typename TT1, typename TT2, typename... TTArgs>
__host__ __device__ bool same_extents(const TT1& A1, const TT2& A2, const TTArgs&... AArgs) {
   if(A1.dimension() != A2.dimension()) {
      return false;
   }
   for(index_t d = 0; d < A1.dimension(); ++d) {
      if(A1.extent(d) != A2.extent(d)) {
         return false;
      }
   }
   if constexpr(sizeof...(TTArgs) == 0) {
      return true;
   } else {
      return same_extents(A2, AArgs...);
   }
}


template <typename TT, typename... TTArgs>
__host__ __device__ constexpr bool same_value_type() {
   return (std::is_same_v<ValueTypeOf<TT>, ValueTypeOf<TTArgs>> && ...);
}


template <typename TT, typename... TTArgs>
__host__ __device__ constexpr bool same_memory_kind() {
   return ((TT::memory_kind() == TTArgs::memory_kind()) && ...);
}


enum Scheme { Host, Device, Unified };


namespace internal {


struct BaseTag {};


template <typename D>
constexpr bool base_tag_v = std::is_base_of_v<BaseTag, D>;


}  // namespace internal


template <typename T, index_t dim, Scheme scheme, bool is_const_ptr = false,
          bool is_pinned_mem = false>
class TensorBase : protected internal::BaseTag {
private:
   using Self = TensorBase<T, dim, scheme, is_const_ptr, is_pinned_mem>;

   using cnd_ptr_t = std::conditional_t<is_const_ptr, const T*, T*>;
   using cnd_ref_t = std::conditional_t<is_const_ptr, const T&, T&>;

   using const_ptr_t = const T*;
   using const_ref_t = const T&;

protected:
   Extents<dim> ext_{};
   cnd_ptr_t data_{};


   __host__ __device__ static inline void validate_host_type() {
#ifdef __CUDA_ARCH__
      if constexpr(scheme == Host) {
         __device__ void not_callable_on_device_error();
         not_callable_on_device_error();
      }
#endif
   }


   __host__ __device__ static inline void validate_device_type() {
#ifndef __CUDA_ARCH__
      if constexpr(scheme != Host) {
         __host__ void not_callable_on_host_error();
         not_callable_on_host_error();
      }
#endif
   }


   __host__ __device__ static inline void validate_host_device_type() {
      validate_host_type();
      validate_device_type();
   }


   __host__ __device__ static inline void validate_host_debug() {
#ifndef NDEBUG
      Self::validate_host_type();
#endif
   }


   __host__ __device__ static inline void validate_host_device_debug() {
#ifndef NDEBUG
      Self::validate_host_device_type();
#endif
   }


   template <typename... Ints>
   __host__ __device__ static constexpr void tensor_static_assert_dimension() {
      static_assert(static_cast<index_t>(sizeof...(Ints)) == Self::dimension(),
                    "index dimension must be equal to the dimension of the tensor");
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
      tensor_static_assert_dimension<Ints...>();
      return data_[this->index_of(indexes...)];
   }


   template <typename... Ints>
   __host__ __device__ const_ref_t operator()(Ints... indexes) const {
      tensor_static_assert_dimension<Ints...>();
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
         // For device or unified memory type.
         ASSERT_CUDA(cudaMemset(this->data(), val, this->size() * sizeof(T)));
      }
   }


   __host__ void memset_async(int val, cudaStream_t stream = 0) {
      static_assert(!this->is_host());
      ASSERT_CUDA(cudaMemsetAsync(this->data(), val, this->size() * sizeof(T), stream));
   }


   template <typename TT>
   __host__ void copy(const TT& A) {
      static_assert(std::is_same_v<value_type, ValueTypeOf<TT>>);
      assert(same_extents(*this, A));
      ASSERT_CUDA(cudaMemcpy(this->data(), A.data(), this->bytes(), cudaMemcpyDefault));
   }


   template <typename TT>
   __host__ void copy_async(const TT& A, cudaStream_t stream = 0) {
      // Host to host copy is not asynchronous.
      static_assert(!(this->is_host() && TT::is_host()));
      static_assert(std::is_same_v<value_type, ValueTypeOf<TT>>);
      assert(same_extents(*this, A));

      // Any host tensor must have pinned memory because copy_async is currently not allowed
      // for host-to-host.
      if constexpr(this->is_host()) {
         static_assert(this->is_pinned());
      }
      if constexpr(TT::is_host()) {
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
class TensorBaseValidated : public TensorBase<T, dim, scheme, is_const_ptr, is_pinned_mem> {
private:
   using Base = TensorBase<T, dim, scheme, is_const_ptr, is_pinned_mem>;

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
      this->validate_host_debug();
      return Base::data();
   }


   __host__ __device__ decltype(auto) data() const {
      this->validate_host_debug();
      return Base::data();
   }


   __host__ __device__ bool empty() const {
      this->validate_host_debug();
      return Base::empty();
   }


   __host__ __device__ decltype(auto) begin() {
      this->validate_host_device_debug();
      return Base::begin();
   }


   __host__ __device__ decltype(auto) begin() const {
      this->validate_host_device_debug();
      return Base::begin();
   }


   __host__ __device__ decltype(auto) cbegin() const {
      this->validate_host_device_debug();
      return Base::cbegin();
   }


   __host__ __device__ decltype(auto) end() {
      this->validate_host_device_debug();
      return Base::end();
   }


   __host__ __device__ decltype(auto) end() const {
      this->validate_host_device_debug();
      return Base::end();
   }


   __host__ __device__ decltype(auto) cend() const {
      this->validate_host_device_debug();
      return Base::cend();
   }


   ////////////////////////////////////////////////////////////////////////////////////////////////
   __host__ __device__ index_t size() const {
      this->validate_host_debug();
      return Base::size();
   }


   __host__ __device__ index_t bytes() const {
      this->validate_host_debug();
      return Base::bytes();
   }


   __host__ __device__ index_t extent(index_t d) const {
      this->validate_host_debug();
      return Base::extent(d);
   }


   __host__ __device__ Extents<dim> extents() const {
      this->validate_host_debug();
      return Base::extents();
   }


   ////////////////////////////////////////////////////////////////////////////////////////////////
   __host__ __device__ bool valid_index(index_t i, index_t d) const {
      this->validate_host_debug();
      return Base::valid_index(i, d);
   }


   template <index_t in_dim, typename First, typename... Ints>
   __host__ __device__ index_t offset_of(First first, Ints... indexes) const {
      this->validate_host_debug();
      return Base::template offset_of<in_dim>(first, indexes...);
   }


   template <typename First, typename... Ints>
   __host__ __device__ index_t index_of(First first, Ints... indexes) const {
      this->validate_host_debug();
      return Base::index_of(first, indexes...);
   }


   template <typename... Ints>
   __host__ __device__ decltype(auto) operator()(Ints... indexes) {
      this->validate_host_device_debug();
      return Base::operator()(indexes...);
   }


   template <typename... Ints>
   __host__ __device__ decltype(auto) operator()(Ints... indexes) const {
      this->validate_host_device_debug();
      return Base::operator()(indexes...);
   }


   template <typename Int>
   __host__ __device__ decltype(auto) operator[](Int i) {
      this->validate_host_device_debug();
      return Base::operator[](i);
   }


   template <typename Int>
   __host__ __device__ decltype(auto) operator[](Int i) const {
      this->validate_host_device_debug();
      return Base::operator[](i);
   }
};


}  // namespace tnb
