#include "container.h"

#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace demo {
namespace {

constexpr std::uint8_t kMagic[8] = {0x89, 'J', 'B', 'A', '\r', '\n', 0x1a, '\n'};

std::size_t dtype_size(DType t) {
  switch (t) {
    case DType::U8: return 1;
    case DType::I32: return 4;
    case DType::I64: return 8;
    case DType::F32: return 4;
    case DType::F64: return 8;
  }
  return 0;
}

template <class T>
constexpr DType dtype_of();
template <> constexpr DType dtype_of<std::uint8_t>() { return DType::U8; }
template <> constexpr DType dtype_of<std::int32_t>() { return DType::I32; }
template <> constexpr DType dtype_of<std::int64_t>() { return DType::I64; }
template <> constexpr DType dtype_of<float>() { return DType::F32; }
template <> constexpr DType dtype_of<double>() { return DType::F64; }

std::size_t pad8(std::size_t n) { return (n + 7) / 8 * 8; }

}  // namespace

std::uint64_t Section::count() const {
  std::uint64_t n = 1;
  for (std::uint64_t d : shape) n *= d;
  return n;
}

template <class T>
std::vector<T> Section::as() const {
  if (dtype != dtype_of<T>()) throw std::runtime_error("a section has an unexpected data type");
  std::vector<T> v(count());
  if (!v.empty()) std::memcpy(v.data(), bytes.data(), v.size() * sizeof(T));
  return v;
}

template <class T>
Section make_section(const std::vector<T>& values, std::vector<std::uint64_t> shape) {
  Section s;
  s.dtype = dtype_of<T>();
  s.shape = std::move(shape);
  if (s.count() != values.size()) throw std::logic_error("make_section: the shape does not match the values");
  s.bytes.resize(values.size() * sizeof(T));
  if (!values.empty()) std::memcpy(s.bytes.data(), values.data(), s.bytes.size());
  return s;
}

#define DEMO_INSTANTIATE(T)                                                    \
  template std::vector<T> Section::as<T>() const;                              \
  template Section make_section<T>(const std::vector<T>&, std::vector<std::uint64_t>);
DEMO_INSTANTIATE(std::uint8_t)
DEMO_INSTANTIATE(std::int32_t)
DEMO_INSTANTIATE(std::int64_t)
DEMO_INSTANTIATE(float)
DEMO_INSTANTIATE(double)
#undef DEMO_INSTANTIATE

std::map<std::string, Section> read_sections(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  const std::vector<std::uint8_t> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  std::size_t pos = 0;
  auto need = [&](std::size_t n) {
    if (data.size() - pos < n) throw std::runtime_error(path + ": truncated");
  };
  auto u32 = [&] {
    need(4);
    std::uint32_t v;
    std::memcpy(&v, &data[pos], 4);
    pos += 4;
    return v;
  };
  need(16);
  if (std::memcmp(data.data(), kMagic, 8) != 0) throw std::runtime_error(path + ": not a file of this format");
  pos = 8;
  if (u32() != 1) throw std::runtime_error(path + ": unknown version");
  const std::uint32_t n_sections = u32();
  std::map<std::string, Section> out;
  for (std::uint32_t k = 0; k < n_sections; ++k) {
    const std::uint32_t name_len = u32();
    if (name_len > 256) throw std::runtime_error(path + ": a section name is too long");
    need(name_len + 4);
    std::string name(reinterpret_cast<const char*>(&data[pos]), name_len);
    pos += name_len;
    Section s;
    s.dtype = static_cast<DType>(data[pos]);
    const std::uint8_t ndim = data[pos + 1];
    pos += 4;
    if (dtype_size(s.dtype) == 0) throw std::runtime_error(path + ": unknown data type in " + name);
    need(8u * ndim);
    s.shape.resize(ndim);
    if (ndim) std::memcpy(s.shape.data(), &data[pos], 8u * ndim);
    pos = pad8(pos + 8u * ndim);
    const std::uint64_t bytes = s.count() * dtype_size(s.dtype);
    need(static_cast<std::size_t>(bytes));
    s.bytes.assign(data.begin() + static_cast<std::ptrdiff_t>(pos), data.begin() + static_cast<std::ptrdiff_t>(pos + bytes));
    pos = pad8(pos + static_cast<std::size_t>(bytes));
    if (pos > data.size()) throw std::runtime_error(path + ": truncated");
    if (!out.emplace(std::move(name), std::move(s)).second) throw std::runtime_error(path + ": a section appears twice");
  }
  if (pos != data.size()) throw std::runtime_error(path + ": data after the last section");
  return out;
}

void write_sections(const std::string& path, const std::vector<std::pair<std::string, Section>>& sections) {
  std::vector<std::uint8_t> out(kMagic, kMagic + 8);
  auto put = [&](const void* p, std::size_t n) {
    const auto* b = static_cast<const std::uint8_t*>(p);
    out.insert(out.end(), b, b + n);
  };
  auto pad = [&] { out.resize(pad8(out.size()), 0); };
  const std::uint32_t version = 1, count = static_cast<std::uint32_t>(sections.size());
  put(&version, 4);
  put(&count, 4);
  for (const auto& [name, s] : sections) {
    const std::uint32_t len = static_cast<std::uint32_t>(name.size());
    put(&len, 4);
    put(name.data(), name.size());
    const std::uint8_t head[4] = {static_cast<std::uint8_t>(s.dtype), static_cast<std::uint8_t>(s.shape.size()), 0, 0};
    put(head, 4);
    put(s.shape.data(), 8 * s.shape.size());
    pad();
    put(s.bytes.data(), s.bytes.size());
    pad();
  }
  std::ofstream f(path, std::ios::binary);
  if (!f || !f.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size())))
    throw std::runtime_error("cannot write " + path);
}

}  // namespace demo
