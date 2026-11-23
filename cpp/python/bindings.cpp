// Python bindings (nanobind) for the otrv C++ core. Built by scikit-build-core as otrv_native._core.
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <exception>
#include <system_error>
#include <variant>

#include "otrv/sei.hpp"

namespace nb = nanobind;

namespace {

nb::object to_py(const otrv::sei::Value& v) {
  return std::visit(
      [](auto x) -> nb::object {
        using T = decltype(x);
        if constexpr (std::is_same_v<T, bool>)
          return nb::bool_(x);
        else if constexpr (std::is_same_v<T, double>)
          return nb::float_(x);
        else
          return nb::int_(x);
      },
      v);
}

// The wire fields of one frame, keys in wire order (tsplat.sei.decode_proto's shape).
nb::dict fields_to_py(const otrv::sei::Frame& f) {
  nb::dict d;
  for (const auto& [num, v] : f.fields()) {
    const auto name = otrv::sei::kFields[static_cast<std::size_t>(num)].name;
    d[nb::str(name.data(), name.size())] = to_py(v);
  }
  return d;
}

// Same shape as tsplat.sei.extract_py: the wire fields plus gear/autopilot labels.
nb::list frames_to_py(const std::vector<otrv::sei::Frame>& frames) {
  nb::list out;
  for (const auto& f : frames) {
    nb::dict d = fields_to_py(f);
    const auto gear = f.gear();
    const auto ap = f.autopilot();
    d["gear"] = nb::str(gear.data(), gear.size());
    d["autopilot"] = nb::str(ap.data(), ap.size());
    out.append(d);
  }
  return out;
}

}  // namespace

NB_MODULE(_core, m) {
  m.doc() = "C++ core of Open Tesla Reality View";

  nb::register_exception_translator([](const std::exception_ptr& p, void*) {
    try {
      std::rethrow_exception(p);
    } catch (const otrv::sei::Mp4Error& e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
    } catch (const std::system_error& e) {
      PyErr_SetString(PyExc_OSError, e.what());
    }
  });

  m.def(
      "sei_extract",
      [](const std::string& path) {
        std::vector<otrv::sei::Frame> frames;
        {
          nb::gil_scoped_release release;  // decoding a minute of video takes a few ms; let other threads run
          frames = otrv::sei::extract_file(path);
        }
        return frames_to_py(frames);
      },
      nb::arg("path"), "Decode a TeslaCam clip's SEI telemetry. Same output as tsplat.sei.extract_py.");

  m.def(
      "sei_extract_bytes",
      [](nb::bytes data) {
        const auto* p = static_cast<const std::uint8_t*>(data.data());
        return frames_to_py(otrv::sei::extract({p, data.size()}));
      },
      nb::arg("data"), "Like sei_extract, for an MP4 already in memory.");

  m.def(
      "sei_decode_proto",
      [](nb::bytes data) -> nb::object {
        const auto* p = static_cast<const std::uint8_t*>(data.data());
        const auto frame = otrv::sei::decode_proto({p, data.size()});
        if (!frame) return nb::none();
        return fields_to_py(*frame);
      },
      nb::arg("data"), "Decode one SeiMetadata protobuf. Same output as tsplat.sei.decode_proto.");
}
