// Georeferencing a reconstruction with the car's own GPS.
//
// Structure-from-motion recovers the drive up to an unknown similarity: scale, orientation and
// position are arbitrary. Every SEI frame carries the car's latitude and longitude, so the
// reconstructed trajectory can be fitted to the GPS track. That gives metric scale, true north and
// a geographic origin in one solve, and the scene can be put on a map.
//
// The model is the one that suits a ground vehicle with no GPS altitude (SEI has none): the scene's
// "up" (from the camera rig) is rotated onto local vertical first, then a 2D similarity
// (scale, yaw, east/north offset) is fitted in the horizontal plane. That stays well-posed on a
// straight road, where a full 7-DoF fit would leave the roll about the road undetermined. The fit
// is RANSAC (2-point minimal samples) followed by least squares on the inliers, so GPS multipath
// jumps and tunnel dropouts don't drag the solution.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace otrv::geo {

using Vec3 = std::array<double, 3>;
using Mat3 = std::array<std::array<double, 3>, 3>;

struct Geodetic {
  double lat_deg = 0, lon_deg = 0, height_m = 0;
};

// WGS84 geodetic <-> Earth-centred, Earth-fixed (metres).
[[nodiscard]] Vec3 to_ecef(const Geodetic& g);
[[nodiscard]] Geodetic from_ecef(const Vec3& ecef);

// Local east-north-up frame (metres) tangent to the ellipsoid at `origin`.
[[nodiscard]] Vec3 to_enu(const Geodetic& g, const Geodetic& origin);
[[nodiscard]] Geodetic from_enu(const Vec3& enu, const Geodetic& origin);

// A rotation taking unit vector `from` to unit vector `to` (shortest arc; any valid rotation when
// they are opposite).
[[nodiscard]] Mat3 rotation_between(Vec3 from, Vec3 to);

struct PathSample {
  double t;  // seconds since the start of the window
  Vec3 p;    // reconstructed position, scene units
};

struct GpsSample {
  double t;
  double lat_deg, lon_deg;
};

struct FitOptions {
  double inlier_m = 3.0;     // RANSAC inlier threshold on the horizontal residual
  double max_gap_s = 2.0;    // don't interpolate GPS across a longer dropout
  double min_extent_m = 15;  // the drive must cover at least this much ground
  int ransac_iters = 1000;   // used when there are too many pairs to try them all
  std::uint64_t seed = 0x5eed;
};

// scene -> ENU: x_enu = scale * Rz(yaw) * level * p + translation
struct Georef {
  bool ok = false;
  std::string reason;  // why the fit was rejected (empty when ok)

  Geodetic origin;                  // ENU origin: the first GPS fix in the window
  Mat3 level{};                     // rotates the scene's up onto +z
  double scale = 0;                 // metres per scene unit
  double yaw_rad = 0;               // rotation about +z after levelling
  Vec3 translation{};               // metres, ENU
  Mat3 rotation{};                  // Rz(yaw) * level
  std::array<double, 16> matrix{};  // the full transform, row-major 4x4

  std::size_t matched = 0;  // path samples with an interpolated GPS position
  std::size_t inliers = 0;
  double rms_m = 0;     // horizontal residual over the inliers
  double max_m = 0;     // largest inlier residual
  double extent_m = 0;  // diagonal of the GPS track's bounding box

  [[nodiscard]] Vec3 to_enu(const Vec3& scene) const;
  [[nodiscard]] Geodetic to_geodetic(const Vec3& scene) const;
};

// Fits the path to the GPS track. `up` is the scene's up direction (path.json "up"). GPS samples
// with no fix (0, 0) are ignored, and a run of identical fixes counts once, at its first sample:
// the car repeats the last fix in every frame until the receiver reports a new one.
[[nodiscard]] Georef fit(const std::vector<PathSample>& path, const Vec3& up, const std::vector<GpsSample>& gps,
                         const FitOptions& opts = {});

}  // namespace otrv::geo
