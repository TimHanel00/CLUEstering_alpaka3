
#pragma once


#include "CLUEstering/data_structures/AssociationMapView.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/internal/alpaka/memory.hpp"

#include <span>
#include <cstdint>
#include <cstddef>
#include <utility>
#include <stdexcept>

namespace clue {
  template<typename TDev>
  class Followers;

  namespace internal {
    template<std::size_t Ndim, typename TDev>
    class Tiles;

    template <typename TQueue>
    auto make_associator(TQueue&, std::span<const int32_t>, int32_t); // forward decl with non-auto return if needed
  }

  namespace detail {
    /// @brief The AssociationMap class is a data structure that maps keys to values.
  /// It associates integer keys with integer values in ono-to-many or many-to-many associations.
  ///
  /// @tparam TDev The device type to use for the allocation. Defaults to clue::Device.
   template <typename TDev>
  class AssociationMapBase
  {
  public:
    using key_type    = int32_t;
    using mapped_type = int32_t;
    using size_type   = std::size_t;

    using iterator       = mapped_type*;
    using const_iterator = mapped_type const*;

    using keys_container_type   = getBufferType<TDev, key_type>;
    using mapped_container_type = getBufferType<TDev, mapped_type>;

    struct Extents { size_type keys{}, values{}; };

    struct Containers {
      keys_container_type const& keys;
      mapped_container_type const& values;
    };
    AssociationMapBase() = delete;


    auto size() const { return m_extents.keys; }
    auto extents() const { return m_extents; }

    iterator begin() { return m_indexes.data(); }
    const_iterator begin() const { return m_indexes.data(); }
    const_iterator cbegin() const { return m_indexes.data(); }

    iterator end() { return m_indexes.data() + m_offsets[m_extents.keys]; }
    const_iterator end() const { return m_indexes.data() + m_offsets[m_extents.keys]; }
    const_iterator cend() const { return m_indexes.data() + m_offsets[m_extents.keys]; }

    size_type count(key_type key) const {
      if(key < 0 || key >= static_cast<key_type>(m_extents.keys))
        throw std::out_of_range("AssociationMap::count key out of range");
      return m_offsets[key + 1] - m_offsets[key];
    }

    bool contains(key_type key) const {
      if(key < 0 || key >= static_cast<key_type>(m_extents.keys))
        throw std::out_of_range("AssociationMap::contains key out of range");
      return m_offsets[key + 1] > m_offsets[key];
    }

    iterator lower_bound(key_type key) {
      if(key < 0 || key >= static_cast<key_type>(m_extents.keys))
        throw std::out_of_range("AssociationMap::lower_bound key out of range");
      return m_indexes.data() + m_offsets[key];
    }
    const_iterator lower_bound(key_type key) const {
      if(key < 0 || key >= static_cast<key_type>(m_extents.keys))
        throw std::out_of_range("AssociationMap::lower_bound key out of range");
      return m_indexes.data() + m_offsets[key];
    }

    iterator upper_bound(key_type key) {
      if(key < 0 || key >= static_cast<key_type>(m_extents.keys))
        throw std::out_of_range("AssociationMap::upper_bound key out of range");
      return m_indexes.data() + m_offsets[key + 1];
    }
    const_iterator upper_bound(key_type key) const {
      if(key < 0 || key >= static_cast<key_type>(m_extents.keys))
        throw std::out_of_range("AssociationMap::upper_bound key out of range");
      return m_indexes.data() + m_offsets[key + 1];
    }

    std::pair<iterator, iterator> equal_range(key_type key) {
      return {lower_bound(key), upper_bound(key)};
    }
    std::pair<const_iterator, const_iterator> equal_range(key_type key) const {
      return {lower_bound(key), upper_bound(key)};
    }

    Containers extract() const { return Containers{m_offsets, m_indexes}; }

    AssociationMapView const& view() const { return m_view; }
    AssociationMapView& view() { return m_view; }

  protected:
     AssociationMapBase(mapped_container_type indexes, keys_container_type offsets)
       : m_indexes(std::move(indexes))
       , m_offsets(std::move(offsets))
     , m_extents{} {}
    // called by derived after buffers are allocated
    void wire_view(size_type nelements, size_type nbins)
    {
      if(nelements == 0)
        throw std::invalid_argument("AssociationMap: nelements must be > 0");

      m_extents = {nbins, nelements};

      m_view.m_indexes = m_indexes.data();
      m_view.m_offsets = m_offsets.data();
      m_view.m_extents = {nbins, nelements};
    }

    void reset(size_type nelements, size_type nbins)
    {
      m_extents = {nbins, nelements};
      m_view.m_extents = {nbins, nelements};
    }

    // accessors used by friends / derived
    keys_container_type& offsets_buf() { return m_offsets; }
    mapped_container_type& indexes_buf() { return m_indexes; }
    keys_container_type const& offsets_buf() const { return m_offsets; }
    mapped_container_type const& indexes_buf() const { return m_indexes; }
    mapped_container_type m_indexes;
    keys_container_type   m_offsets;
    AssociationMapView    m_view;
    Extents              m_extents;

    friend class ::clue::Followers<TDev>;


    template <std::size_t, class>
    friend class ::clue::internal::Tiles;

    template <typename _TQueue>
    friend auto ::clue::internal::make_associator(_TQueue&, std::span<const int32_t>, int32_t);
  };

  } // namespace detail
} // namespace clue
