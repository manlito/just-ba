// Named, typed arrays in the container of the problem file format (document 11):
// the same header, section layout and alignment, so the same readers (numpy's
// frombuffer included) work on any file written here. The demo's capture file is one.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace demo {

enum class DType : std::uint8_t { U8 = 1, I32 = 2, I64 = 3, F32 = 4, F64 = 5 };

struct Section {
  DType dtype = DType::U8;
  std::vector<std::uint64_t> shape;
  std::vector<std::uint8_t> bytes;

  std::uint64_t count() const;
  template <class T>
  std::vector<T> as() const;  // a copy of the data; T must match dtype
};

template <class T>
Section make_section(const std::vector<T>& values, std::vector<std::uint64_t> shape);

// Throws std::runtime_error on a damaged file (wrong magic or version, truncated or
// trailing data, an unknown dtype, a repeated name).
std::map<std::string, Section> read_sections(const std::string& path);
// Sections in the order given. Throws std::runtime_error.
void write_sections(const std::string& path, const std::vector<std::pair<std::string, Section>>& sections);

}  // namespace demo
