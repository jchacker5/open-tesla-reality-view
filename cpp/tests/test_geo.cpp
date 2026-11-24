#include <doctest/doctest.h>

#include <cmath>
#include <numbers>
#include <random>

#include "otrv/geo.hpp"

namespace geo = otrv::geo;
using geo::Vec3;

namespace {

constexpr double kPi = std::numbers::pi;
const geo::Geodetic kOrigin{41.7015, -71.155, 0};  // a street in Fall River, MA

double dist(const Vec3& a, const Vec3& b) { return std::hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]); }

Vec3 mat_vec(const geo::Mat3& m, const Vec3& v) {
  return {m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2], m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
          m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2]};
}

geo::Mat3 transpose(const geo::Mat3& m) {
  geo::Mat3 t{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) t[i][j] = m[j][i];
  return t;
}

// A drive in ENU metres at ~12.5 m/s (28 mph): `straight` seconds east-north-east, then a 90 degree
// left turn of radius 25 m, then on. Flat road, height 0.
Vec3 drive(double t, double straight = 8) {
  const double v = 12.5, h0 = 0.35;  // heading, radians from east
  if (t <= straight) return {v * t * std::cos(h0), v * t * std::sin(h0), 0};
  const Vec3 c0{v * straight * std::cos(h0), v * straight * std::sin(h0), 0};
  const double r = 25, turn = (t - straight) * v / r;
  if (turn <= kPi / 2) {
    const Vec3 centre{c0[0] - r * std::sin(h0), c0[1] + r * std::cos(h0), 0};
    const double a = h0 - kPi / 2 + turn;
    return {centre[0] + r * std::cos(a), centre[1] + r * std::sin(a), 0};
  }
  const double extra = (t - straight - (kPi / 2) * r / v) * v, h1 = h0 + kPi / 2;
  const Vec3 centre{c0[0] - r * std::sin(h0), c0[1] + r * std::cos(h0), 0};
  const Vec3 c1{centre[0] + r * std::cos(h0), centre[1] + r * std::sin(h0), 0};
  return {c1[0] + extra * std::cos(h1), c1[1] + extra * std::sin(h1), 0};
}

// What SfM would give us: the same path under an arbitrary similarity (scale, a rotation that
// also tilts "up", an offset), plus the scene up vector the rig gives.
struct Scene {
  double units_per_m = 0.37;
  geo::Mat3 r;  // ENU -> scene rotation
  Vec3 offset{4, -2, 7};
  Scene() {
    // up tilted 60 degrees towards some arbitrary direction, plus a yaw
    const Vec3 up = {std::sin(1.0) * std::cos(0.4), std::sin(1.0) * std::sin(0.4), std::cos(1.0)};
    const geo::Mat3 tilt = geo::rotation_between({0, 0, 1}, up);
    const double y = 2.1;
    const geo::Mat3 yaw{{{std::cos(y), -std::sin(y), 0}, {std::sin(y), std::cos(y), 0}, {0, 0, 1}}};
    geo::Mat3 m{};
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k) m[i][j] += tilt[i][k] * yaw[k][j];
    r = m;
  }
  [[nodiscard]] Vec3 from_enu(const Vec3& e) const {
    const Vec3 q = mat_vec(r, e);
    return {units_per_m * q[0] + offset[0], units_per_m * q[1] + offset[1], units_per_m * q[2] + offset[2]};
  }
  [[nodiscard]] Vec3 up() const { return mat_vec(r, {0, 0, 1}); }
};

struct Track {
  std::vector<geo::PathSample> path;
  std::vector<geo::GpsSample> gps;
};

// Path at 6 fps; GPS written every frame at 36 fps but only updated at `gps_hz`, with noise and
// occasional multipath jumps.
Track make_track(double seconds, double noise_m, double outlier_rate, double gps_hz = 10, double straight = 8,
                 std::uint64_t seed = 1) {
  Scene scene;
  Track tr;
  for (int i = 0; i <= static_cast<int>(seconds * 6); ++i) {
    const double t = i / 6.0;
    tr.path.push_back({t, scene.from_enu(drive(t, straight))});
  }
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> n(0, noise_m);
  std::uniform_real_distribution<double> u(0, 1);
  geo::GpsSample last{};
  double next_update = 0;
  for (int i = 0; i <= static_cast<int>(seconds * 36); ++i) {
    const double t = i / 36.0;
    if (t >= next_update - 1e-9) {
      Vec3 e = drive(t, straight);
      e[0] += n(rng), e[1] += n(rng);
      if (u(rng) < outlier_rate) e[0] += 40, e[1] -= 25;
      const auto g = geo::from_enu(e, kOrigin);
      last = {t, g.lat_deg, g.lon_deg};
      next_update += 1.0 / gps_hz;
    }
    tr.gps.push_back({t, last.lat_deg, last.lon_deg});
  }
  return tr;
}

// Worst error between the fitted ENU path and the truth.
double worst_error(const geo::Georef& g, const Track& tr, double straight = 8) {
  const Vec3 origin_shift = geo::to_enu(kOrigin, g.origin);
  double worst = 0;
  for (const auto& s : tr.path) {
    Vec3 truth = drive(s.t, straight);
    for (int k = 0; k < 3; ++k) truth[static_cast<std::size_t>(k)] += origin_shift[static_cast<std::size_t>(k)];
    const Vec3 got = g.to_enu(s.p);
    worst = std::max(worst, std::hypot(got[0] - truth[0], got[1] - truth[1]));
  }
  return worst;
}

}  // namespace

TEST_CASE("WGS84 round trips") {
  for (const geo::Geodetic g :
       {geo::Geodetic{0, 0, 0}, geo::Geodetic{41.7015, -71.155, 12}, geo::Geodetic{-33.9, 151.2, 50},
        geo::Geodetic{89.9, 10, 0}, geo::Geodetic{-60, -179.9, 3000}}) {
    const auto back = geo::from_ecef(geo::to_ecef(g));
    CHECK(back.lat_deg == doctest::Approx(g.lat_deg).epsilon(1e-12));
    CHECK(back.lon_deg == doctest::Approx(g.lon_deg).epsilon(1e-12));
    CHECK(back.height_m == doctest::Approx(g.height_m).epsilon(1e-6).scale(1));
  }
  // The equator at 0, 0 is (a, 0, 0).
  CHECK(geo::to_ecef({0, 0, 0})[0] == doctest::Approx(6378137.0));
}

TEST_CASE("ENU") {
  // 0.001 degree of latitude at 41.7 N is about 111.08 m north.
  const Vec3 n = geo::to_enu({kOrigin.lat_deg + 0.001, kOrigin.lon_deg, 0}, kOrigin);
  CHECK(n[0] == doctest::Approx(0).scale(1));
  CHECK(n[1] == doctest::Approx(111.08).epsilon(1e-3));
  // ...and 0.001 degree of longitude N cos(lat) pi/180e3 = 83.24 m east.
  const Vec3 e = geo::to_enu({kOrigin.lat_deg, kOrigin.lon_deg + 0.001, 0}, kOrigin);
  CHECK(e[0] == doctest::Approx(83.237).epsilon(1e-4));
  for (const Vec3 p : {Vec3{120, -45, 3}, Vec3{-2000, 1500, -10}}) {
    CHECK(dist(geo::to_enu(geo::from_enu(p, kOrigin), kOrigin), p) < 1e-6);
  }
}

TEST_CASE("rotation_between") {
  for (const auto& [from, to] : {std::pair<Vec3, Vec3>{{0, 0, 1}, {1, 0, 0}},
                                 {{0.3, -0.2, 0.9}, {-0.5, 0.5, 0.1}},
                                 {{0, 1, 0}, {0, 1, 0}},
                                 {{0, 0, 1}, {0, 0, -1}},
                                 {{1, 2, 3}, {-1, -2, -3}}}) {
    const auto r = geo::rotation_between(from, to);
    const double nf = std::hypot(from[0], from[1], from[2]), nt = std::hypot(to[0], to[1], to[2]);
    const Vec3 got = mat_vec(r, {from[0] / nf, from[1] / nf, from[2] / nf});
    CHECK(dist(got, {to[0] / nt, to[1] / nt, to[2] / nt}) < 1e-12);
    const auto rt = transpose(r);  // orthonormal with determinant +1
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        double s = 0;
        for (int k = 0; k < 3; ++k) s += r[i][k] * rt[k][j];
        CHECK(s == doctest::Approx(i == j ? 1.0 : 0.0).scale(1));
      }
    const double det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]) -
                       r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0]) +
                       r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
    CHECK(det == doctest::Approx(1.0));
  }
}

TEST_CASE("fit recovers the transform from clean GPS") {
  const Scene scene;
  const auto tr = make_track(20, 0.0, 0.0);
  const auto g = geo::fit(tr.path, scene.up(), tr.gps);
  REQUIRE_MESSAGE(g.ok, g.reason);
  CHECK(g.scale == doctest::Approx(1.0 / scene.units_per_m).epsilon(1e-3));
  CHECK(g.inliers == g.matched);
  CHECK(g.rms_m < 0.5);  // 10 Hz GPS, linear interpolation along a 25 m turn
  CHECK(worst_error(g, tr) < 1.0);
  // The scene up vector maps to ENU up.
  const Vec3 up = mat_vec(g.rotation, scene.up());
  CHECK(up[2] == doctest::Approx(1.0));
  // The geodetic origin is the first fix, and the first path point lands on it.
  const auto first = g.to_geodetic(tr.path.front().p);
  CHECK(first.lat_deg == doctest::Approx(kOrigin.lat_deg).epsilon(1e-9));
  CHECK(first.lon_deg == doctest::Approx(kOrigin.lon_deg).epsilon(1e-9));
}

TEST_CASE("fit survives receiver noise and multipath jumps") {
  const Scene scene;
  const auto tr = make_track(20, 1.0, 0.1, 5, 8, 42);
  const auto g = geo::fit(tr.path, scene.up(), tr.gps);
  REQUIRE_MESSAGE(g.ok, g.reason);
  CHECK(g.scale == doctest::Approx(1.0 / scene.units_per_m).epsilon(0.02));
  CHECK(g.inliers < g.matched);  // the jumps were rejected...
  CHECK(g.inliers * 10 > g.matched * 7);
  CHECK(g.rms_m < 2.0);
  CHECK(worst_error(g, tr) < 2.0);  // ...and didn't pull the fit
}

TEST_CASE("a straight road is enough") {
  const Scene scene;
  const auto tr = make_track(12, 0.5, 0.0, 10, 1e9);
  const auto g = geo::fit(tr.path, scene.up(), tr.gps);
  REQUIRE_MESSAGE(g.ok, g.reason);
  CHECK(g.scale == doctest::Approx(1.0 / scene.units_per_m).epsilon(0.01));
  CHECK(worst_error(g, tr, 1e9) < 1.5);
}

TEST_CASE("fit rejects what it can't solve") {
  const Scene scene;
  SUBCASE("no fixes") {
    auto tr = make_track(10, 0, 0);
    for (auto& s : tr.gps) s.lat_deg = s.lon_deg = 0;
    const auto g = geo::fit(tr.path, scene.up(), tr.gps);
    CHECK_FALSE(g.ok);
    CHECK(g.reason.find("GPS fixes") != std::string::npos);
  }
  SUBCASE("parked") {
    auto tr = make_track(10, 0, 0);
    for (auto& s : tr.gps) s.lat_deg = kOrigin.lat_deg + 1e-6 * s.t, s.lon_deg = kOrigin.lon_deg;
    CHECK(geo::fit(tr.path, scene.up(), tr.gps).reason.find("moved too little") != std::string::npos);
  }
  SUBCASE("path and GPS don't overlap in time") {
    auto tr = make_track(10, 0, 0);
    for (auto& s : tr.path) s.t += 100;
    CHECK(geo::fit(tr.path, scene.up(), tr.gps).reason.find("overlap") != std::string::npos);
  }
  SUBCASE("long dropouts aren't bridged") {
    auto tr = make_track(20, 0, 0, 0.27);  // one fix every 3.7 s, never on a path sample
    CHECK_FALSE(geo::fit(tr.path, scene.up(), tr.gps).ok);
    geo::FitOptions o;
    o.max_gap_s = 5;
    CHECK(geo::fit(tr.path, scene.up(), tr.gps, o).ok);
  }
  SUBCASE("a reconstruction that doesn't match the GPS at all") {
    auto tr = make_track(20, 0, 0);
    std::mt19937_64 rng(3);
    std::uniform_real_distribution<double> u(-50, 50);
    for (auto& s : tr.path) s.p = {u(rng), u(rng), u(rng)};
    CHECK_FALSE(geo::fit(tr.path, scene.up(), tr.gps).ok);
  }
  SUBCASE("zero up vector") {
    CHECK_FALSE(geo::fit(make_track(10, 0, 0).path, {0, 0, 0}, make_track(10, 0, 0).gps).ok);
  }
}

TEST_CASE("matrix and to_enu agree") {
  const Scene scene;
  const auto tr = make_track(20, 0.3, 0.0);
  const auto g = geo::fit(tr.path, scene.up(), tr.gps);
  REQUIRE(g.ok);
  for (const auto& s : tr.path) {
    const Vec3 a = g.to_enu(s.p);
    Vec3 b{};
    for (std::size_t i = 0; i < 3; ++i)
      b[i] =
          g.matrix[4 * i] * s.p[0] + g.matrix[4 * i + 1] * s.p[1] + g.matrix[4 * i + 2] * s.p[2] + g.matrix[4 * i + 3];
    CHECK(dist(a, b) < 1e-9);
  }
}
