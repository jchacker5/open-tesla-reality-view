#include "otrv/sei.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <system_error>

namespace otrv::sei {
namespace {

constexpr std::uint8_t kNalTypeSei = 6;
constexpr std::uint8_t kSeiUserDataUnregistered = 5;
constexpr std::uint8_t kUuidPad = 0x42;
constexpr std::uint8_t kUuidEnd = 0x69;

// Little-endian loads that work regardless of host byte order or alignment.
template <typename T>
T load_le(const std::uint8_t* p) {
  using U = std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;
  U u = 0;
  for (std::size_t k = 0; k < sizeof(T); ++k) u |= static_cast<U>(p[k]) << (8 * k);
  return std::bit_cast<T>(u);
}

std::uint32_t load_be32(const std::uint8_t* p) {
  return (std::uint32_t{p[0]} << 24) | (std::uint32_t{p[1]} << 16) | (std::uint32_t{p[2]} << 8) | std::uint32_t{p[3]};
}

std::uint64_t load_be64(const std::uint8_t* p) { return (std::uint64_t{load_be32(p)} << 32) | load_be32(p + 4); }

// Reads a base-128 varint at buf[i]. nullopt if it runs off the end or exceeds 64 bits.
std::optional<std::uint64_t> read_varint(std::span<const std::uint8_t> buf, std::size_t& i) {
  std::uint64_t result = 0;
  for (int shift = 0; shift < 64; shift += 7) {
    if (i >= buf.size()) return std::nullopt;
    const std::uint8_t b = buf[i++];
    const std::uint64_t chunk = b & 0x7F;
    // The 10th byte (shift 63) may only contribute the top bit.
    if (shift == 63 && chunk > 1) return std::nullopt;
    result |= chunk << shift;
    if (!(b & 0x80)) return result;
  }
  return std::nullopt;
}

bool truthy(const Value& v) {
  return std::visit([](auto x) { return x != decltype(x){}; }, v);
}

// Integer value of an enum field; nullopt if it arrived with a non-varint wire type. (Enum fields
// are never `bool` kind, so a bool can't reach here.)
std::optional<std::uint64_t> as_index(const Value* v) {
  if (v == nullptr) return 0;  // proto3 default
  if (const auto* u = std::get_if<std::uint64_t>(v)) return *u;
  return std::nullopt;
}

template <std::size_t N>
std::string_view lookup(const std::array<std::string_view, N>& table, const Value* v) {
  const auto idx = as_index(v);
  return idx && *idx < N ? table[*idx] : std::string_view{"?"};
}

std::optional<double> as_double(const Value* v) {
  if (v == nullptr) return std::nullopt;
  if (const auto* d = std::get_if<double>(v)) return *d;
  return std::nullopt;
}

}  // namespace

void Frame::set(int field_number, Value v) {
  for (auto& [num, val] : fields_) {
    if (num == field_number) {
      val = v;
      return;
    }
  }
  fields_.emplace_back(field_number, v);
}

const Value* Frame::get(int field_number) const {
  for (const auto& [num, val] : fields_)
    if (num == field_number) return &val;
  return nullptr;
}

const Value* Frame::get(std::string_view name) const {
  for (std::size_t n = 1; n < kFields.size(); ++n)
    if (kFields[n].name == name) return get(static_cast<int>(n));
  return nullptr;
}

std::optional<std::uint64_t> Frame::frame_seq_no() const {
  const Value* v = get(3);
  if (v == nullptr) return std::nullopt;
  if (const auto* u = std::get_if<std::uint64_t>(v)) return *u;
  return std::nullopt;
}

std::optional<double> Frame::speed_mps() const { return as_double(get(4)); }
std::optional<double> Frame::latitude_deg() const { return as_double(get(11)); }
std::optional<double> Frame::longitude_deg() const { return as_double(get(12)); }
std::string_view Frame::gear() const { return lookup(kGears, get(2)); }
std::string_view Frame::autopilot() const { return lookup(kAutopilot, get(10)); }

std::optional<Frame> decode_proto(std::span<const std::uint8_t> buf) {
  Frame out;
  std::size_t i = 0;
  while (i < buf.size()) {
    const auto key = read_varint(buf, i);
    if (!key) return std::nullopt;
    const std::uint64_t num = *key >> 3;
    const unsigned wire = static_cast<unsigned>(*key & 7);
    Value val;
    switch (wire) {
      case 0: {
        const auto v = read_varint(buf, i);
        if (!v) return std::nullopt;
        val = *v;
        break;
      }
      case 1:
        if (buf.size() - i < 8) return std::nullopt;
        val = load_le<double>(buf.data() + i);
        i += 8;
        break;
      case 5:
        if (buf.size() - i < 4) return std::nullopt;
        val = static_cast<double>(load_le<float>(buf.data() + i));
        i += 4;
        break;
      case 2: {
        const auto len = read_varint(buf, i);
        if (!len) return std::nullopt;
        // A length past the end just ends the message, as in the Python decoder.
        i = *len > buf.size() - i ? buf.size() : i + static_cast<std::size_t>(*len);
        continue;
      }
      default:
        return std::nullopt;  // groups (3, 4) and reserved wire types
    }
    if (num >= 1 && num < kFields.size()) {
      const int n = static_cast<int>(num);
      out.set(n, kFields[num].kind == Kind::Bool ? Value{truthy(val)} : val);
    }
  }
  if (out.empty()) return std::nullopt;
  return out;
}

std::vector<std::uint8_t> strip_emulation_prevention(std::span<const std::uint8_t> data) {
  std::vector<std::uint8_t> out;
  out.reserve(data.size());
  int zeros = 0;
  for (const std::uint8_t b : data) {
    if (zeros >= 2 && b == 3) {
      zeros = 0;
      continue;
    }
    out.push_back(b);
    zeros = b == 0 ? zeros + 1 : 0;
  }
  return out;
}

std::optional<std::vector<std::uint8_t>> sei_payload(std::span<const std::uint8_t> nal) {
  // nal[0] NAL header, nal[1] SEI payload type, nal[2] payload size, then the UUID.
  for (std::size_t i = 3; i + 1 < nal.size(); ++i) {
    if (nal[i] == kUuidPad) continue;
    if (nal[i] != kUuidEnd) break;
    return strip_emulation_prevention(nal.subspan(i + 1, nal.size() - i - 2));
  }
  return std::nullopt;
}

namespace {

// Random-access byte sources for the MP4 walk. read() returns how many bytes it could read, so a
// read past the end behaves like a short read past EOF in the Python decoder.
class SpanSource {
 public:
  explicit SpanSource(std::span<const std::uint8_t> data) : data_(data) {}
  [[nodiscard]] std::uint64_t size() const { return data_.size(); }
  std::size_t read(std::uint64_t off, std::size_t n, std::uint8_t* out) const {
    if (off >= data_.size()) return 0;
    const std::size_t got = std::min<std::uint64_t>(n, data_.size() - off);
    std::memcpy(out, data_.data() + off, got);
    return got;
  }

 private:
  std::span<const std::uint8_t> data_;
};

// Reads only the bytes the walk needs: atom headers, NAL length prefixes and SEI NAL units. A 4 KiB
// window makes the length prefix, SEI message and next slice header one read; the video slices in
// between (tens of KB each at 1280x960) are skipped, so a clip is never read whole. 4 KiB measured
// fastest against 16 and 64 KiB on a full-resolution clip.
class FileSource {
 public:
  explicit FileSource(const std::filesystem::path& path) : in_(path, std::ios::binary), path_(path) {
    if (!in_) throw std::system_error(errno, std::generic_category(), "cannot open " + path.string());
    size_ = std::filesystem::file_size(path);
    window_.resize(kWindow);
  }
  [[nodiscard]] std::uint64_t size() const { return size_; }
  std::size_t read(std::uint64_t off, std::size_t n, std::uint8_t* out) {
    std::size_t done = 0;
    while (done < n && off + done < size_) {
      const std::uint64_t at = off + done;
      if (at < win_off_ || at >= win_off_ + win_len_) fill(at);
      const auto from = static_cast<std::size_t>(at - win_off_);
      const std::size_t take = std::min(n - done, win_len_ - from);
      std::memcpy(out + done, window_.data() + from, take);
      done += take;
    }
    return done;
  }

 private:
  static constexpr std::size_t kWindow = 4 * 1024;

  void fill(std::uint64_t at) {
    const auto want = static_cast<std::size_t>(std::min<std::uint64_t>(kWindow, size_ - at));
    in_.clear();
    in_.seekg(static_cast<std::streamoff>(at));
    in_.read(reinterpret_cast<char*>(window_.data()), static_cast<std::streamsize>(want));
    if (static_cast<std::size_t>(in_.gcount()) != want)
      throw std::system_error(EIO, std::generic_category(), "cannot read " + path_.string());
    win_off_ = at;
    win_len_ = want;
  }

  std::ifstream in_;
  std::filesystem::path path_;
  std::uint64_t size_ = 0;
  std::vector<std::uint8_t> window_;
  std::uint64_t win_off_ = 0;
  std::size_t win_len_ = 0;
};

template <typename Source>
std::vector<Frame> extract_from(Source& src) {
  std::uint8_t hdr[16];

  // Find the top-level mdat atom.
  std::uint64_t pos = 0;
  std::uint64_t mdat_start = 0;
  std::uint64_t mdat_size = 0;  // 0 = runs to the end of the file
  for (;;) {
    if (src.read(pos, 8, hdr) < 8) throw Mp4Error("mdat atom not found");
    pos += 8;
    const std::uint32_t size32 = load_be32(hdr);
    const bool is_mdat = std::memcmp(hdr + 4, "mdat", 4) == 0;
    std::uint64_t size = size32;
    std::uint64_t header = 8;
    if (size32 == 1) {
      if (src.read(pos, 8, hdr + 8) < 8) throw Mp4Error("truncated MP4 atom header");
      size = load_be64(hdr + 8);
      header = 16;
      pos += 8;
    }
    if (is_mdat) {
      if (size != 0 && size < header) return {};  // an mdat smaller than its own header holds nothing
      mdat_start = pos;
      mdat_size = size != 0 ? size - header : 0;
      break;
    }
    if (size < header) throw Mp4Error("invalid MP4 atom size");
    const std::uint64_t skip = size - header;
    pos = skip > src.size() ? src.size() + 1 : pos + skip;  // saturate; the next read then fails
  }

  // Walk the length-prefixed (AVCC) NAL units.
  std::vector<Frame> rows;
  std::vector<std::uint8_t> nal;
  std::uint64_t at = mdat_start;
  std::uint64_t used = 0;
  const auto advance = [&](std::uint64_t n) { at = n > src.size() ? src.size() : std::min(at + n, src.size()); };
  while (mdat_size == 0 || used < mdat_size) {
    if (src.read(at, 4, hdr) < 4) break;
    at += 4;
    const std::uint32_t n = load_be32(hdr);
    if (n < 2) {
      advance(n);
      used += 4 + std::uint64_t{n};
      continue;
    }
    if (src.read(at, 2, hdr) < 2) break;
    used += 4 + std::uint64_t{n};
    if ((hdr[0] & 0x1F) != kNalTypeSei || hdr[1] != kSeiUserDataUnregistered) {
      advance(n);
      continue;
    }
    // Size the buffer by what's left in the file, not by the (untrusted) length prefix.
    nal.resize(static_cast<std::size_t>(std::min<std::uint64_t>(n, src.size() - std::min(at, src.size()))));
    nal.resize(src.read(at, nal.size(), nal.data()));
    advance(n);
    const auto payload = sei_payload(nal);
    if (!payload) continue;
    auto frame = decode_proto(*payload);
    if (frame && frame->get(3) != nullptr) rows.push_back(std::move(*frame));
  }
  return rows;
}

}  // namespace

std::vector<Frame> extract(std::span<const std::uint8_t> mp4) {
  SpanSource src(mp4);
  return extract_from(src);
}

std::vector<Frame> extract_file(const std::filesystem::path& path) {
  FileSource src(path);
  return extract_from(src);
}

std::string python_float_repr(double v) {
  if (std::isnan(v)) return "NaN";
  if (std::isinf(v)) return v > 0 ? "Infinity" : "-Infinity";
  if (v == 0.0) return std::signbit(v) ? "-0.0" : "0.0";

  // Shortest round-trip digits in scientific form: [-]d[.ddd]e[+-]XX
  char sci[64];
  const auto res = std::to_chars(sci, sci + sizeof sci, v, std::chars_format::scientific);
  std::string_view s(sci, static_cast<std::size_t>(res.ptr - sci));
  const bool neg = s.front() == '-';
  if (neg) s.remove_prefix(1);
  const std::size_t e = s.find('e');
  std::string digits;
  for (const char c : s.substr(0, e))
    if (c != '.') digits.push_back(c);
  int exp = 0;
  std::from_chars(s.data() + e + 1 + (s[e + 1] == '+'), s.data() + s.size(), exp);

  std::string out = neg ? "-" : "";
  if (exp < -4 || exp >= 16) {
    out += digits[0];
    if (digits.size() > 1) out += "." + digits.substr(1);
    out += exp < 0 ? "e-" : "e+";
    const int a = exp < 0 ? -exp : exp;
    if (a < 10) out += '0';
    out += std::to_string(a);
  } else if (exp < 0) {
    out += "0." + std::string(static_cast<std::size_t>(-exp - 1), '0') + digits;
  } else {
    const auto int_len = static_cast<std::size_t>(exp + 1);
    if (digits.size() <= int_len) {
      out += digits + std::string(int_len - digits.size(), '0') + ".0";
    } else {
      out += digits.substr(0, int_len) + "." + digits.substr(int_len);
    }
  }
  return out;
}

std::string to_json(const std::vector<Frame>& frames) {
  if (frames.empty()) return "[]";
  const auto value = [](const Value& v) {
    return std::visit(
        [](auto x) -> std::string {
          using T = decltype(x);
          if constexpr (std::is_same_v<T, bool>)
            return x ? "true" : "false";
          else if constexpr (std::is_same_v<T, double>)
            return python_float_repr(x);
          else
            return std::to_string(x);
        },
        v);
  };
  std::string out = "[\n";
  for (std::size_t r = 0; r < frames.size(); ++r) {
    out += " {\n";
    for (const auto& [num, v] : frames[r].fields()) {
      out += "  \"";
      out += kFields[static_cast<std::size_t>(num)].name;
      out += "\": " + value(v) + ",\n";
    }
    out += "  \"gear\": \"";
    out += frames[r].gear();
    out += "\",\n  \"autopilot\": \"";
    out += frames[r].autopilot();
    out += "\"\n }";
    out += r + 1 < frames.size() ? ",\n" : "\n";
  }
  out += "]";
  return out;
}

}  // namespace otrv::sei
