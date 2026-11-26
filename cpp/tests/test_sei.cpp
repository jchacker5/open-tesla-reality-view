#include <doctest/doctest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "otrv/sei.hpp"

using Bytes = std::vector<std::uint8_t>;
namespace sei = otrv::sei;

namespace {

// ---- fixture builders (mirror tests/test_sei.py) ----

Bytes varint(std::uint64_t n) {
  Bytes out;
  do {
    std::uint8_t b = n & 0x7F;
    n >>= 7;
    out.push_back(n != 0 ? (b | 0x80) : b);
  } while (n != 0);
  return out;
}

Bytes key(int num, int wire) { return varint(static_cast<std::uint64_t>(num) << 3 | static_cast<std::uint64_t>(wire)); }

void append(Bytes& a, const Bytes& b) { a.insert(a.end(), b.begin(), b.end()); }

template <typename T>
Bytes le(T v) {
  Bytes out(sizeof(T));
  std::memcpy(out.data(), &v, sizeof(T));
  if constexpr (std::endian::native == std::endian::big) std::reverse(out.begin(), out.end());
  return out;
}

Bytes be32(std::uint32_t v) {
  return {static_cast<std::uint8_t>(v >> 24), static_cast<std::uint8_t>(v >> 16), static_cast<std::uint8_t>(v >> 8),
          static_cast<std::uint8_t>(v)};
}

struct Telemetry {
  std::uint64_t gear, seq;
  float speed;
  bool brake;
  double lat, lon;
};

Bytes proto(const Telemetry& t) {
  Bytes p;
  append(p, key(2, 0)), append(p, varint(t.gear));
  append(p, key(3, 0)), append(p, varint(t.seq));
  append(p, key(4, 5)), append(p, le(t.speed));
  append(p, key(9, 0)), append(p, varint(t.brake ? 1 : 0));
  append(p, key(11, 1)), append(p, le(t.lat));
  append(p, key(12, 1)), append(p, le(t.lon));
  append(p, key(13, 1)), append(p, le(0.0));  // eight zero bytes: needs emulation prevention
  return p;
}

Bytes add_epb(const Bytes& data) {
  Bytes out;
  int zeros = 0;
  for (const auto b : data) {
    if (zeros >= 2 && b <= 3) {
      out.push_back(3);
      zeros = 0;
    }
    out.push_back(b);
    zeros = b == 0 ? zeros + 1 : 0;
  }
  return out;
}

Bytes sei_nal(const Bytes& payload) {
  Bytes body(17, 0x42);  // UUID padding, then Tesla's 0x69 marker
  body[16] = 0x69;
  append(body, add_epb(payload));
  Bytes nal{0x06, 0x05, static_cast<std::uint8_t>(std::min<std::size_t>(body.size(), 255))};
  append(nal, body);
  nal.push_back(0x80);
  return nal;
}

Bytes mp4(const std::vector<Bytes>& nals) {
  Bytes data;
  for (const auto& n : nals) append(data, be32(static_cast<std::uint32_t>(n.size()))), append(data, n);
  Bytes out = be32(16);
  append(out, Bytes{'f', 't', 'y', 'p', 'i', 's', 'o', 'm', 0, 0, 2, 0});
  append(out, be32(static_cast<std::uint32_t>(8 + data.size())));
  append(out, Bytes{'m', 'd', 'a', 't'});
  append(out, data);
  return out;
}

template <typename T>
T as(const sei::Value* v) {
  REQUIRE(v != nullptr);
  REQUIRE(std::holds_alternative<T>(*v));
  return std::get<T>(*v);
}

}  // namespace

TEST_CASE("extract decodes Tesla SEI messages and skips other NAL units") {
  const Bytes p1 = proto({1, 100, 12.5f, true, 37.7749, -122.4194});
  const Bytes p2 = proto({2, 101, 0.0f, false, 37.775, -122.42});
  REQUIRE(add_epb(p1) != p1);  // the fixture really exercises emulation-prevention stripping

  const Bytes idr{0x65, 0x88, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  const Bytes other_sei{0x06, 0x04, 0x01, 0x02, 0x03};  // SEI, but not user_data_unregistered
  const auto rows = sei::extract(mp4({idr, sei_nal(p1), other_sei, sei_nal(p2)}));

  REQUIRE(rows.size() == 2);
  const auto& r = rows[0];
  CHECK(r.frame_seq_no() == 100);
  CHECK(r.gear() == "D");
  CHECK(as<std::uint64_t>(r.get("gear_state")) == 1);
  CHECK(as<double>(r.get("vehicle_speed_mps")) == doctest::Approx(12.5));
  CHECK(as<bool>(r.get("brake_applied")) == true);
  CHECK(r.latitude_deg() == 37.7749);
  CHECK(r.longitude_deg() == -122.4194);
  CHECK(as<double>(r.get("heading_deg")) == 0.0);
  CHECK(r.autopilot() == "NONE");
  CHECK(rows[1].frame_seq_no() == 101);
  CHECK(rows[1].gear() == "R");
  CHECK(as<bool>(rows[1].get("brake_applied")) == false);
}

TEST_CASE("emulation prevention bytes are removed") {
  CHECK(sei::strip_emulation_prevention(Bytes{0, 0, 3, 1, 0, 0, 3, 0, 0, 3}) == Bytes{0, 0, 1, 0, 0, 0, 0});
  CHECK(sei::strip_emulation_prevention(Bytes{0, 3, 0, 3}) == Bytes{0, 3, 0, 3});  // a single zero isn't a prefix
}

TEST_CASE("varints") {
  Bytes max = key(3, 0);
  append(max, varint(std::numeric_limits<std::uint64_t>::max()));
  CHECK(max.size() == 11);
  const auto f = sei::decode_proto(max);
  REQUIRE(f);
  CHECK(f->frame_seq_no() == std::numeric_limits<std::uint64_t>::max());

  SUBCASE("more than 64 bits is invalid") {
    Bytes over = key(3, 0);
    append(over, Bytes{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x02});
    CHECK_FALSE(sei::decode_proto(over));
  }
  SUBCASE("eleven bytes is invalid") {
    Bytes eleven = key(3, 0);
    append(eleven, Bytes(10, 0x80));
    eleven.push_back(0);
    CHECK_FALSE(sei::decode_proto(eleven));
  }
  SUBCASE("truncated") { CHECK_FALSE(sei::decode_proto(Bytes{0x18, 0x80})); }
}

TEST_CASE("decode_proto field rules") {
  SUBCASE("truncated fixed-width fields fail the message") {
    Bytes p = key(11, 1);
    append(p, Bytes{1, 2, 3});
    CHECK_FALSE(sei::decode_proto(p));
    Bytes q = key(4, 5);
    append(q, Bytes{1, 2});
    CHECK_FALSE(sei::decode_proto(q));
  }
  SUBCASE("length-delimited fields are skipped; a length past the end ends the message") {
    Bytes p = key(3, 0);
    append(p, varint(7));
    append(p, key(20, 2)), append(p, varint(3)), append(p, Bytes{9, 9, 9});
    append(p, key(1, 0)), append(p, varint(2));
    append(p, key(21, 2)), append(p, varint(1000));
    const auto f = sei::decode_proto(p);
    REQUIRE(f);
    CHECK(f->fields().size() == 2);
    CHECK(as<std::uint64_t>(f->get("version")) == 2);
  }
  SUBCASE("group and reserved wire types are invalid") {
    CHECK_FALSE(sei::decode_proto(key(3, 3)));
    CHECK_FALSE(sei::decode_proto(key(3, 6)));
  }
  SUBCASE("unknown fields only means no message") {
    Bytes p = key(99, 0);
    append(p, varint(1));
    CHECK_FALSE(sei::decode_proto(p));
    CHECK_FALSE(sei::decode_proto(Bytes{}));
  }
  SUBCASE("bool fields coerce any wire type; numeric fields keep the wire type") {
    Bytes p = key(7, 5);
    append(p, le(0.5f));
    append(p, key(8, 1)), append(p, le(0.0));
    append(p, key(4, 0)), append(p, varint(3));  // speed sent as a varint
    const auto f = sei::decode_proto(p);
    REQUIRE(f);
    CHECK(as<bool>(f->get("blinker_on_left")) == true);
    CHECK(as<bool>(f->get("blinker_on_right")) == false);
    CHECK(as<std::uint64_t>(f->get("vehicle_speed_mps")) == 3);
    CHECK_FALSE(f->speed_mps());
  }
  SUBCASE("a repeated field overwrites in place and keeps its position") {
    Bytes p = key(3, 0);
    append(p, varint(1));
    append(p, key(1, 0)), append(p, varint(5));
    append(p, key(3, 0)), append(p, varint(2));
    const auto f = sei::decode_proto(p);
    REQUIRE(f);
    REQUIRE(f->fields().size() == 2);
    CHECK(f->fields()[0].first == 3);
    CHECK(f->frame_seq_no() == 2);
  }
  SUBCASE("enum labels") {
    Bytes p = key(3, 0);
    append(p, varint(1));
    CHECK(sei::decode_proto(p)->gear() == "P");  // missing gear_state reads as 0
    append(p, key(2, 0)), append(p, varint(9));
    append(p, key(10, 0)), append(p, varint(3));
    CHECK(sei::decode_proto(p)->gear() == "?");
    CHECK(sei::decode_proto(p)->autopilot() == "TACC");
    append(p, key(2, 1)), append(p, le(1.0));
    CHECK(sei::decode_proto(p)->gear() == "?");  // not an integer
  }
}

TEST_CASE("MP4 container handling") {
  const Bytes nal = sei_nal(proto({1, 7, 3.0f, false, 1.0, 2.0}));

  SUBCASE("no mdat") {
    Bytes f = be32(16);
    append(f, Bytes{'f', 't', 'y', 'p', 'i', 's', 'o', 'm', 0, 0, 2, 0});
    CHECK_THROWS_AS((void)sei::extract(f), sei::Mp4Error);
    CHECK_THROWS_AS((void)sei::extract(Bytes{}), sei::Mp4Error);
  }
  SUBCASE("invalid atom size") {
    Bytes f = be32(4);
    append(f, Bytes{'f', 'r', 'e', 'e'});
    CHECK_THROWS_AS((void)sei::extract(f), sei::Mp4Error);
  }
  SUBCASE("64-bit mdat size") {
    Bytes data = be32(static_cast<std::uint32_t>(nal.size()));
    append(data, nal);
    Bytes f = be32(1);
    append(f, Bytes{'m', 'd', 'a', 't'});
    const std::uint64_t size = 16 + data.size();
    for (int s = 56; s >= 0; s -= 8) f.push_back(static_cast<std::uint8_t>(size >> s));
    append(f, data);
    CHECK(sei::extract(f).size() == 1);
  }
  SUBCASE("truncated 64-bit header") {
    Bytes f = be32(1);
    append(f, Bytes{'m', 'd', 'a', 't', 0, 0});
    CHECK_THROWS_AS((void)sei::extract(f), sei::Mp4Error);
  }
  SUBCASE("mdat size 0 runs to the end of the file") {
    Bytes f = be32(0);
    append(f, Bytes{'m', 'd', 'a', 't'});
    append(f, be32(static_cast<std::uint32_t>(nal.size()))), append(f, nal);
    append(f, be32(static_cast<std::uint32_t>(nal.size()))), append(f, nal);
    CHECK(sei::extract(f).size() == 2);
  }
  SUBCASE("empty and truncated mdat never consume following bytes") {
    for (const std::uint32_t payload_size : {0u, 2u, 4u, 12u}) {
      Bytes f = be32(8 + payload_size);
      append(f, Bytes{'m', 'd', 'a', 't'});
      append(f, be32(static_cast<std::uint32_t>(nal.size())));
      append(f, nal);
      CHECK(sei::extract(f).empty());
    }
  }
  SUBCASE("tiny NAL lengths and a NAL cut off by EOF") {
    Bytes f = mp4({Bytes{0x06}, nal});
    f.resize(f.size() - 3);
    CHECK(sei::extract(f).empty());  // the truncated SEI loses its last proto bytes
  }
  SUBCASE("SEI without frame_seq_no is dropped") {
    Bytes p = key(1, 0);
    append(p, varint(1));
    CHECK(sei::extract(mp4({sei_nal(p)})).empty());
  }
}

TEST_CASE("python_float_repr matches Python's repr") {
  const std::vector<std::pair<double, std::string>> cases{
      {12.5, "12.5"},
      {0.1, "0.1"},
      {-122.4194, "-122.4194"},
      {37.7749, "37.7749"},
      {1e-05, "1e-05"},
      {0.0001, "0.0001"},
      {0.00012345, "0.00012345"},
      {1e16, "1e+16"},
      {1234567890123456.0, "1234567890123456.0"},
      {1e22, "1e+22"},
      {100.0, "100.0"},
      {-0.0, "-0.0"},
      {0.0, "0.0"},
      {5e-324, "5e-324"},
      {1.7976931348623157e308, "1.7976931348623157e+308"},
      {static_cast<double>(12.3f), "12.300000190734863"},
      {std::numeric_limits<double>::quiet_NaN(), "NaN"},
      {-std::numeric_limits<double>::infinity(), "-Infinity"},
  };
  for (const auto& [v, want] : cases) {
    CAPTURE(want);
    CHECK(sei::python_float_repr(v) == want);
  }
}

TEST_CASE("to_json matches json.dumps(rows, indent=1)") {
  const auto rows = sei::extract(mp4({sei_nal(proto({1, 100, 12.5f, true, 37.7749, -122.4194}))}));
  CHECK(sei::to_json(rows) ==
        "[\n {\n  \"gear_state\": 1,\n  \"frame_seq_no\": 100,\n  \"vehicle_speed_mps\": 12.5,\n"
        "  \"brake_applied\": true,\n  \"latitude_deg\": 37.7749,\n  \"longitude_deg\": -122.4194,\n"
        "  \"heading_deg\": 0.0,\n  \"gear\": \"D\",\n  \"autopilot\": \"NONE\"\n }\n]");
  CHECK(sei::to_json({}) == "[]");
}

TEST_CASE("random input never crashes (run under ASan/UBSan in CI)") {
  std::mt19937_64 rng(1234);
  std::uniform_int_distribution<int> byte(0, 255), len(0, 96);
  const Bytes good = sei_nal(proto({1, 1, 1.0f, true, 1.0, 1.0}));
  for (int iter = 0; iter < 20000; ++iter) {
    Bytes buf(static_cast<std::size_t>(len(rng)));
    for (auto& b : buf) b = static_cast<std::uint8_t>(byte(rng));
    (void)sei::decode_proto(buf);
    (void)sei::sei_payload(buf);
    // Mutate a valid file so the walk gets deep into mdat before hitting garbage.
    Bytes f = mp4({good, buf, good});
    f[static_cast<std::size_t>(byte(rng)) % f.size()] = static_cast<std::uint8_t>(byte(rng));
    try {
      (void)sei::extract(f);
    } catch (const sei::Mp4Error&) {
    }
  }
}
