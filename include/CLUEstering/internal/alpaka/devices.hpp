
#pragma once

#include <cassert>
#include <vector>

#include <alpaka/alpaka.hpp>

namespace clue {

  namespace internal {
    struct SelectedBackendAndExec {
#ifdef alpaka_SELECT_CpuSerial
      using apis = std::tuple<alpaka::api::Host>;
      using selectedExec = alpaka::exec::CpuSerial;
#else
#ifdef alpaka_SELECT_OMP
      using apis = std::tuple<alpaka::api::Host>;
      using selectedExec = alpaka::exec::CpuOmpBlocks;
#else
#ifdef alpaka_SELECT_CUDA
      using apis = std::tuple<alpaka::api::Cuda>;
      using selectedExec = alpaka::exec::GpuCuda;
#else
#ifdef alpaka_SELECT_TBB
      using apis = std::tuple<alpaka::api::Host>;
      using selectedExec = alpaka::exec::CpuTbbBlocks;
#else
#ifdef alpaka_SELECT_HIP
      using apis = std::tuple<alpaka::api::Hip>;
      using selectedExec = alpaka::exec::GpuHip;
#else
      /***
            * If no target_compile_definition is specified we simply use the first enabled
            * executor for the API it is enabled for.
            * --> this allows backend selection solely based on the alpaka cmake options
            */
      using apis = ALPAKA_TYPEOF(alpaka::onHost::enabledApis);
      using selectedExec = std::tuple<>;
#endif
#endif
#endif
#endif
#endif
    };
    template <typename T_DevSelector, typename T_Exec>
    class DevicePoolImpl {
    public:
      using Selector = T_DevSelector;
      using Exec = T_Exec;
      using Device = ALPAKA_TYPEOF(std::declval<Selector&>().makeDevice(0U));

      using HostDevice = ALPAKA_TYPEOF(alpaka::onHost::makeHostDevice());

      // --- host device (constant) ---
      static auto hostDev() -> HostDevice & {
        static HostDevice hd = alpaka::onHost::makeHostDevice();
        return hd;
      }

      // --- executor storage ---
      auto exec() -> auto const& { return m_exec; }
      auto devices()-> std::vector<Device> const& { return m_devices; }

      [[nodiscard]] auto size() -> std::size_t { return m_devices.size(); }

      auto deviceAt(std::size_t idx) -> Device & { return m_devices.at(idx); }

      auto indexOf(Device const& dev) const -> std::size_t {
        for (std::size_t i = 0; i < m_devices.size(); ++i) {
          if (m_devices[i] == dev) {
            return i;
          }
        }
        throw std::runtime_error("clue::DevicePool: device not found");
      }
      explicit DevicePoolImpl(T_DevSelector selector, T_Exec exec) : m_exec(exec) {
        if(!selector.isAvailable())
        {
          throw std::runtime_error(
            std::string{"No device available!"}
          );
        }
        m_devices.reserve(selector.getDeviceCount());
        for (uint32_t dev_id = 0; dev_id < selector.getDeviceCount(); ++dev_id) {
          m_devices.emplace_back(selector.makeDevice(dev_id));
        }
      }
      T_Exec m_exec;
      std::vector<Device> m_devices;
    };
    // return the alpaka accelerator devices for the given platform
    template <typename T_DeviceSelector>
    auto devices(T_DeviceSelector const& selector) {
      std::vector<ALPAKA_TYPEOF(selector.makeDevice(0U))> devices;
      if(!selector.isAvailable())
      {
        throw std::runtime_error(
          std::string{"No device available!"}
        );
      }
      for (uint32_t dev_id = 0; dev_id < selector.getDeviceCount(); dev_id++) {
        devices.emplace_back(selector.makeDevice(dev_id));
      }
      return devices;
    }
  }  // namespace internal
  struct DevicePool {
    using APIS = internal::SelectedBackendAndExec::apis;
    using Execs = std::conditional_t<
        std::is_same_v<internal::SelectedBackendAndExec::selectedExec, std::tuple<>>,
        ALPAKA_TYPEOF(alpaka::exec::enabledExecutors),
        std::tuple<internal::SelectedBackendAndExec::selectedExec>>;
    static auto& get() {
      auto backend = std::get<0>(alpaka::onHost::allBackends(APIS{}, Execs{}));
      auto device_spec = backend[alpaka::object::deviceSpec];
      auto dev_selector = alpaka::onHost::makeDeviceSelector(device_spec);
      auto exec = backend[alpaka::object::exec];
      static auto dev_pool = internal::DevicePoolImpl{std::move(dev_selector), std::move(exec)};
      return dev_pool;
    }
    // retrieve the host device
    static auto& getHost() { return get().hostDev(); }
    // return the alpaka executor which is used for the current backend
    static auto exec() { return get().exec(); }
    // get the index of a given alpaka device
    static auto indexOf(auto const& device) { return get().indexOf(device); }
    // retrieve a alpaka device from a given index
    static auto& deviceAt(uint32_t idx) { return get().deviceAt(idx); }
    // get the list of devices for the currently selected backend
    static auto& devices() { return get().devices(); }
  };
  using Device = ALPAKA_TYPEOF(DevicePool::deviceAt(0));

}  // namespace clue
