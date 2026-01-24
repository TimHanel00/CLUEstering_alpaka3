/// @file AssociationMap.hpp
/// @brief Provides the AssociationMap class for managing associations between keys and values
/// @authors Simone Balducci, Felice Pantaleo, Marco Rovere, Wahid Redjeb, Aurora Perego, Francesco Giacomini

#pragma once

#include "CLUEstering/data_structures/AssociationMapView.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/internal/alpaka/memory.hpp"
#include "CLUEstering/data_structures/detail/AssociationMapBase.hpp"
#include <span>
#include <alpaka/alpaka.hpp>
#include "CLUEstering/data_structures/detail/AssociationMap.hpp"
namespace clue {
  namespace detail {
    template <typename TDev>
    class AssociationMapBase;
  }
  namespace internal {

    template <std::size_t Ndim, typename TDev>
    class Tiles;
  }  // namespace internal
class HostAssociationMap : public detail::AssociationMapBase<alpaka::api::Host>
  {
    using Base = detail::AssociationMapBase<alpaka::api::Host>;
  public:
    using Base::size_type;
    using Base::key_type;
    using Base::mapped_type;

    HostAssociationMap() = delete;

    HostAssociationMap(size_type nelements, size_type nbins)
      : Base(
          make_host_buffer<mapped_type>(nelements),
          make_host_buffer<key_type>(nbins + 1))
    {
      Base::wire_view(nelements, nbins);
      std::memset(Base::m_offsets.data(), 0, nbins * sizeof(key_type));
    }

    ALPAKA_FN_HOST void initialize(size_type nelements, size_type nbins)
    {
      Base::m_indexes = make_host_buffer<mapped_type>(nelements);
      Base::m_offsets = make_host_buffer<key_type>(nbins + 1);
      Base::wire_view(nelements, nbins);
      std::memset(Base::m_offsets.data(), 0, nbins * sizeof(key_type));
    }
    template <concepts::Queue TQueue>
    ALPAKA_FN_HOST void fill(
    TQueue&,
    size_type size,
    std::span<const key_type> associations)
    {
      // sanity check (optional but recommended)
      ALPAKA_ASSERT(size == associations.size());

      fill(associations);
    }
    ALPAKA_FN_HOST void fill(std::span<const key_type> associations)
    {
      std::vector<key_type> sizes(Base::m_extents.keys, 0);
      std::for_each(associations.begin(), associations.end(), [&](key_type key) {
        if (key >= 0) {
          ++sizes[key];
        }
      });

      std::vector<key_type> temporary_keys(Base::m_extents.keys + 1);
      temporary_keys[0] = 0;
      std::inclusive_scan(sizes.begin(), sizes.end(), temporary_keys.begin() + 1);
      std::copy(temporary_keys.data(), temporary_keys.data() + Base::m_extents.keys + 1, Base::m_offsets.data());
      for (auto i = 0u; i < associations.size(); ++i) {
        if (associations[i] >= 0) {
          auto& offset = temporary_keys[associations[i]];
          Base::m_indexes[offset] = i;
          ++offset;
        }
      }
    }
  };
  template<typename TDev>
  class DevAssociationMap : public detail::AssociationMapBase<TDev>
  {
    using Base = detail::AssociationMapBase<TDev>;
  public:
    using typename Base::size_type;
    using typename Base::key_type;
    using typename Base::mapped_type;

    DevAssociationMap() = delete;

    DevAssociationMap(TDev const& dev, size_type nelements, size_type nbins)
      : Base(
          make_device_buffer<mapped_type>(dev, nelements), //index buffer allocation
          make_device_buffer<key_type>(dev, nbins + 1))    //offset buffer allocation
    {
      Base::wire_view(nelements, nbins);
    }

    template <concepts::Queue TQueue>
    DevAssociationMap(size_type nelements, size_type nbins, TQueue& queue)
      : Base(
          make_device_buffer<mapped_type>(queue, size_type{nelements}), //index buffer allocation
          make_device_buffer<key_type>(queue, size_type{nbins + 1}))  //offset buffer allocation
    {
      Base::wire_view(nelements, nbins);
    }
    template <concepts::Queue TQueue>
    ALPAKA_FN_HOST void initialize(
    TQueue& queue,
    size_type nelements,
    size_type nbins)
    {
      Base::m_indexes = make_device_buffer<mapped_type>(queue.getDevice(), size_type{nelements});
      Base::m_offsets = make_device_buffer<key_type>(queue.getDevice(), size_type{nbins + 1});

      Base::wire_view(nelements, nbins);
    }
    static void dump_range(const char* name, const int32_t* p, int begin, int end) {
      std::cout << name << " [" << begin << ".." << end << "]: ";
      for (int i = begin; i <= end; ++i) {
        std::cout << i << ":" << p[i] << " ";
      }
      std::cout << std::endl;
    }

    static void dump_head(const char* name, const int32_t* p, int n, int count=16) {
      dump_range(name, p, 0, std::min(n-1, count-1));
    }


    template <concepts::Queue TQueue,class TFunc>
    ALPAKA_FN_HOST void fill(TQueue& queue,size_type size, TFunc func)
    {
      auto exec = DevicePool::exec();
      auto device = queue.getDevice();
    if (Base::m_extents.keys == 0)
        return;
    const int32_t nbins = static_cast<int32_t>(Base::m_extents.keys);


    // 1) Build per-element association/bin info
    auto bin_buffer = make_device_buffer<int32_t>(device, std::size_t{size});

    constexpr auto blocksize = size_type{512};
    const auto gridsize = alpaka::divCeil(size, blocksize);
    const auto workdiv  = alpaka::onHost::FrameSpec{gridsize, blocksize};

      queue.enqueue(
          exec,
          workdiv,
          detail::KernelComputeAssociations<TFunc>{},
          size,
          bin_buffer.data(),
          nbins,
          func);

    // 2) Compute per-key sizes (histogram-like)
    auto sizes_buffer = make_device_buffer<int32_t>(device, size_type{Base::m_extents.keys});
    alpaka::onHost::memset(queue, sizes_buffer, 0);

    queue.enqueue(
        exec,
        workdiv,
        detail::KernelComputeAssociationSizes{},
        bin_buffer.data(),
        sizes_buffer.data(),
        nbins,
        size);

    // 3) Prefix scan -> offsets
    //    We want:
    //      temp_offsets[0] = 0
    //      temp_offsets[i+1] = sum_{j<=i} sizes[j]
    //    This is exactly exclusiveScan(sizes) into temp_offsets+1.

    auto temp_offsets = make_device_buffer<int32_t>(device, size_type{Base::m_extents.keys + 1});
    alpaka::onHost::memset(queue, temp_offsets, int32_t{0}, Vec1D{1});

    auto sizes_mdspan   = alpaka::makeMdSpan(sizes_buffer.data(), Vec1D{Base::m_extents.keys});
    auto offsets_mdspan = alpaka::makeMdSpan(temp_offsets.data() + 1, Vec1D{Base::m_extents.keys});

    const auto scanBufferSize =
        alpaka::onHost::getScanBufferSize<int32_t>(sizes_mdspan.getExtents());

    auto scan_buffer = make_device_buffer<std::byte>(device, Vec1D{scanBufferSize});
    alpaka::onHost::inclusiveScan(
        queue,
        exec,
        scan_buffer,
        offsets_mdspan,
        sizes_mdspan);

    // 4) Copy offsets into Base storage
    alpaka::onHost::memcpy(
        queue,
        alpaka::makeView(device, Base::m_offsets.data(), Vec1D{Base::m_extents.keys + 1}),
        temp_offsets);
    // 5) Fill associator indices using computed offsets
    queue.enqueue(
    exec,
        workdiv,
        detail::KernelFillAssociator{},
        Base::m_indexes.data(),
        bin_buffer.data(),
        temp_offsets.data(),nbins,
        size);
      alpaka::onHost::wait(queue);
    }

    template <concepts::Queue TQueue>
    ALPAKA_FN_HOST void fill(TQueue& queue,size_type size, std::span<const key_type> associations)
    {
      auto exec = DevicePool::exec();
      if (Base::m_extents.keys == 0)
            return;
      const int32_t nbins = static_cast<int32_t>(Base::m_extents.keys);
      constexpr size_type blockSize = 512;
      const size_type gridSize = alpaka::divCeil(size, blockSize);
      const auto frameSpec = alpaka::onHost::FrameSpec(gridSize, blockSize);

      auto sizes_buffer = make_device_buffer<key_type>(queue.getDevice(), size_type{Base::m_extents.keys});
      alpaka::onHost::memset(queue, sizes_buffer, 0);
      queue.enqueue(       exec,frameSpec,
                           detail::KernelComputeAssociationSizes{},
                           associations.data(),
                           sizes_buffer.data(),nbins,
                           size);

      auto block_counter = make_device_buffer<int32_t>(queue.getDevice());
      alpaka::onHost::memset(queue, block_counter, 0);

      // Allocate output offsets (size = keys + 1)
      auto temp_offsets = make_device_buffer<key_type>(queue.getDevice(), Base::m_extents.keys + 1);

      // temp_offsets[0] = 0
      alpaka::onHost::memset(queue, temp_offsets, key_type{0}, Vec1D{1});
      // Create mdspans
      auto sizes_mdspan =alpaka::makeMdSpan(sizes_buffer.data(),Vec1D{Base::m_extents.keys});

      auto offsets_mdspan =
          alpaka::makeMdSpan(
              temp_offsets.data() + 1,
              Vec1D{Base::m_extents.keys});


      auto scanBufferSize =
          alpaka::onHost::getScanBufferSize<key_type>(sizes_mdspan.getExtents());

      auto scan_buffer = make_device_buffer<std::byte>(queue.getDevice(), Vec1D{scanBufferSize});

      alpaka::onHost::inclusiveScan(
          queue,
          exec,
          scan_buffer,
          offsets_mdspan,
          sizes_mdspan);


      alpaka::onHost::memcpy(queue,
                     alpaka::makeView(queue.getDevice(), Base::m_offsets.data(), Vec1D{Base::m_extents.keys + 1}),
                     temp_offsets);
      queue.enqueue(exec,frameSpec,
                         detail::KernelFillAssociator{},
                         Base::m_indexes.data(),
                         associations.data(),
                         temp_offsets.data(),
                         nbins,
                         size);
    }
  };
}  // namespace clue
