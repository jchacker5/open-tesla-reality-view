// Python bindings (nanobind) for the otrv C++ core. Built by scikit-build-core as otrv_native._core.
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <exception>
#include <system_error>
#include <variant>

#include "otrv/geo.hpp"
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

nb::dict georef_to_py(const otrv::geo::Georef& g, const std::vector<otrv::geo::PathSample>& path) {
  nb::dict d;
  d["ok"] = g.ok;
  d["reason"] = g.reason;
  d["matched"] = g.matched;
  d["inliers"] = g.inliers;
  d["extent_m"] = g.extent_m;
  if (!g.ok) return d;
  d["origin"] = nb::make_tuple(g.origin.lat_deg, g.origin.lon_deg);
  d["metres_per_unit"] = g.scale;
  d["yaw_deg"] = g.yaw_rad * 180.0 / 3.14159265358979323846;
  d["translation"] = g.translation;
  d["matrix"] = g.matrix;
  d["rms_m"] = g.rms_m;
  d["max_m"] = g.max_m;
  nb::list track;  // where the reconstruction puts each path sample on the map
  for (const auto& s : path) {
    const auto geo = g.to_geodetic(s.p);
    track.append(nb::make_tuple(s.t, geo.lat_deg, geo.lon_deg));
  }
  d["track"] = track;
  return d;
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

  m.def(
      "geo_fit",
      [](const std::vector<std::tuple<double, double, double, double>>& path, const otrv::geo::Vec3& up,
         const std::vector<std::tuple<double, double, double>>& gps, double inlier_m, double max_gap_s,
         double min_extent_m) {
        std::vector<otrv::geo::PathSample> p;
        p.reserve(path.size());
        for (const auto& [t, x, y, z] : path) p.push_back({t, {x, y, z}});
        std::vector<otrv::geo::GpsSample> g;
        g.reserve(gps.size());
        for (const auto& [t, lat, lon] : gps) g.push_back({t, lat, lon});
        otrv::geo::FitOptions o;
        o.inlier_m = inlier_m;
        o.max_gap_s = max_gap_s;
        o.min_extent_m = min_extent_m;
        otrv::geo::Georef r;
        {
          nb::gil_scoped_release release;
          r = otrv::geo::fit(p, up, g, o);
        }
        return georef_to_py(r, p);
      },
      nb::arg("path"), nb::arg("up"), nb::arg("gps"), nb::arg("inlier_m") = 3.0, nb::arg("max_gap_s") = 2.0,
      nb::arg("min_extent_m") = 15.0,
      "Fit a reconstructed path [(t, x, y, z)] to GPS [(t, lat, lon)]: metric scale, heading and origin.");
}
