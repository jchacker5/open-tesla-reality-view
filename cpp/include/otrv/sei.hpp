// Tesla dashcam SEI telemetry decoder.
//
// Every TeslaCam clip recorded on firmware 2025.44.25+ (HW3/HW4) carries one H.264 SEI message
// per frame (NAL type 6, payload type 5, user_data_unregistered) whose payload is a protobuf
// `SeiMetadata` message (teslamotors/dashcam, dashcam.proto): speed, gear, steering, pedals,
// blinkers, Autopilot state, GPS and IMU.
//
// This is a dependency-free C++20 port of tsplat/sei.py. It decodes the same fields with the same
// rules, including how it treats malformed input, so either implementation can stand in for the
// other; tests/test_sei_native.py checks that on fuzzed input.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace otrv::sei {

// A decoded field value. The wire type decides the alternative (varint -> uint64, fixed64 ->
// double, fixed32 -> float widened to double), and fields declared `bool` are converted to bool.
using Value = std::variant<std::uint64_t, double, bool>;

enum class Kind : std::uint8_t { Varint, F32, F64, Bool };

struct FieldSpec {
  std::string_view name;
  Kind kind;
};

// dashcam.proto field numbers 1..16. Index 0 is unused.
inline constexpr std::array<FieldSpec, 17> kFields{{
    {"", Kind::Varint},
    {"version", Kind::Varint},
    {"gear_state", Kind::Varint},
    {"frame_seq_no", Kind::Varint},
    {"vehicle_speed_mps", Kind::F32},
    {"accelerator_pedal_position", Kind::F32},
    {"steering_wheel_angle", Kind::F32},
    {"blinker_on_left", Kind::Bool},
    {"blinker_on_right", Kind::Bool},
    {"brake_applied", Kind::Bool},
    {"autopilot_state", Kind::Varint},
    {"latitude_deg", Kind::F64},
    {"longitude_deg", Kind::F64},
    {"heading_deg", Kind::F64},
    {"linear_acceleration_mps2_x", Kind::F64},
    {"linear_acceleration_mps2_y", Kind::F64},
    {"linear_acceleration_mps2_z", Kind::F64},
}};

inline constexpr std::array<std::string_view, 4> kGears{"P", "D", "R", "N"};
inline constexpr std::array<std::string_view, 4> kAutopilot{"NONE", "SELF_DRIVING", "AUTOSTEER", "TACC"};

// One decoded SeiMetadata message. Fields keep the order they first appeared on the wire (a
// repeated field overwrites its value in place), which is also the key order the Python decoder
// produces.
class Frame {
 public:
  void set(int field_number, Value v);
  [[nodiscard]] const Value* get(int field_number) const;
  [[nodiscard]] const Value* get(std::string_view name) const;
  [[nodiscard]] const std::vector<std::pair<int, Value>>& fields() const { return fields_; }
  [[nodiscard]] bool empty() const { return fields_.empty(); }

  // Convenience accessors; nullopt when the field is absent or has an unexpected wire type.
  [[nodiscard]] std::optional<std::uint64_t> frame_seq_no() const;
  [[nodiscard]] std::optional<double> speed_mps() const;
  [[nodiscard]] std::optional<double> latitude_deg() const;
  [[nodiscard]] std::optional<double> longitude_deg() const;

  // "P"/"D"/"R"/"N" (or "NONE"/"SELF_DRIVING"/...), "?" for an out-of-range or non-integer value.
  // A missing field reads as 0, as in proto3.
  [[nodiscard]] std::string_view gear() const;
  [[nodiscard]] std::string_view autopilot() const;

 private:
  std::vector<std::pair<int, Value>> fields_;
};

// Decodes one SeiMetadata protobuf. Unknown field numbers and length-delimited fields are
// skipped. Returns nullopt for a truncated or malformed message, a varint longer than 64 bits,
// a group wire type, or a message with no known fields.
[[nodiscard]] std::optional<Frame> decode_proto(std::span<const std::uint8_t> buf);

// Removes H.264 emulation-prevention bytes (the 0x03 in every 00 00 03 sequence).
[[nodiscard]] std::vector<std::uint8_t> strip_emulation_prevention(std::span<const std::uint8_t> data);

// Extracts Tesla's protobuf from one SEI NAL unit (starting at the NAL header byte). Tesla pads
// the user_data_unregistered UUID with 0x42 bytes and ends it with 0x69; the payload follows and
// the last byte (RBSP trailing bits) is dropped. nullopt if the NAL doesn't have that shape.
[[nodiscard]] std::optional<std::vector<std::uint8_t>> sei_payload(std::span<const std::uint8_t> nal);

class Mp4Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Walks the MP4's top-level atoms to the `mdat`, reads its length-prefixed (AVCC) NAL units and
// returns one Frame per SEI message that carries a frame_seq_no. Throws Mp4Error when there is no
// mdat or an atom header is invalid.
[[nodiscard]] std::vector<Frame> extract(std::span<const std::uint8_t> mp4);
[[nodiscard]] std::vector<Frame> extract_file(const std::filesystem::path& path);

// JSON array of objects, keys in wire order plus "gear" and "autopilot", formatted like Python's
// json.dumps(rows, indent=1) so the CLI output diffs cleanly against `python -m tsplat.sei`.
[[nodiscard]] std::string to_json(const std::vector<Frame>& frames);

// Python's repr() of a float: shortest round-trip digits, scientific notation when the decimal
// exponent is < -4 or >= 16, "NaN"/"Infinity" as json.dumps writes them.
[[nodiscard]] std::string python_float_repr(double v);

}  // namespace otrv::sei
