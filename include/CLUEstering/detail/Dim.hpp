#pragma once
#include <cstddef>
namespace clue{
  template<size_t N>
  struct Dim{
    constexpr static size_t size = N;
  };

}
