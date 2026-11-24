#include "otrv/geo.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>
#include <random>

namespace otrv::geo {
namespace {

constexpr double kA = 6378137.0;            // WGS84 semi-major axis, m
constexpr double kF = 1.0 / 298.257223563;  // flattening
constexpr double kE2 = kF * (2.0 - kF);     // first eccentricity squared
constexpr double kDeg = std::numbers::pi / 180.0;

using C = std::complex<double>;

Vec3 sub(const Vec3& a, const Vec3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
double dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double norm(const Vec3& a) { return std::sqrt(dot(a, a)); }
Vec3 normalized(const Vec3& a) {
  const double n = norm(a);
  return {a[0] / n, a[1] / n, a[2] / n};
}

Vec3 mul(const Mat3& m, const Vec3& v) { return {dot(m[0], v), dot(m[1], v), dot(m[2], v)}; }

Mat3 mul(const Mat3& a, const Mat3& b) {
  Mat3 r{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) r[i][j] += a[i][k] * b[k][j];
  return r;
}

Mat3 rot_z(double a) {
  const double c = std::cos(a), s = std::sin(a);
  return {{{c, -s, 0}, {s, c, 0}, {0, 0, 1}}};
}

// Rows of the ECEF -> ENU rotation at a geodetic origin.
Mat3 enu_basis(const Geodetic& o) {
  const double sl = std::sin(o.lat_deg * kDeg), cl = std::cos(o.lat_deg * kDeg);
  const double so = std::sin(o.lon_deg * kDeg), co = std::cos(o.lon_deg * kDeg);
  return {{{-so, co, 0}, {-sl * co, -sl * so, cl}, {cl * co, cl * so, sl}}};
}

// 2D similarity b = z * a + t, as complex numbers: z = scale * e^(i yaw).
struct Sim2 {
  C z{1, 0};
  C t{0, 0};
  [[nodiscard]] C apply(C a) const { return z * a + t; }
};

// Least-squares similarity (the 2D Umeyama solution, closed form in complex arithmetic).
std::optional<Sim2> fit_sim2(const std::vector<C>& a, const std::vector<C>& b, const std::vector<std::size_t>& idx) {
  if (idx.size() < 2) return std::nullopt;
  C ma{}, mb{};
  for (const auto i : idx) ma += a[i], mb += b[i];
  ma /= static_cast<double>(idx.size());
  mb /= static_cast<double>(idx.size());
  C num{};
  double den = 0;
  for (const auto i : idx) {
    const C da = a[i] - ma;
    num += std::conj(da) * (b[i] - mb);
    den += std::norm(da);
  }
  if (den <= 1e-18) return std::nullopt;
  const C z = num / den;
  return Sim2{z, mb - z * ma};
}

}  // namespace

Vec3 to_ecef(const Geodetic& g) {
  const double sl = std::sin(g.lat_deg * kDeg), cl = std::cos(g.lat_deg * kDeg);
  const double so = std::sin(g.lon_deg * kDeg), co = std::cos(g.lon_deg * kDeg);
  const double n = kA / std::sqrt(1.0 - kE2 * sl * sl);  // prime vertical radius of curvature
  return {(n + g.height_m) * cl * co, (n + g.height_m) * cl * so, (n * (1.0 - kE2) + g.height_m) * sl};
}

Geodetic from_ecef(const Vec3& e) {
  // Bowring's method with a few fixed-point refinements: sub-millimetre everywhere on Earth.
  const double lon = std::atan2(e[1], e[0]);
  const double p = std::hypot(e[0], e[1]);
  double lat = std::atan2(e[2], p * (1.0 - kE2));
  double h = 0;
  for (int k = 0; k < 6; ++k) {
    const double sl = std::sin(lat);
    const double n = kA / std::sqrt(1.0 - kE2 * sl * sl);
    h = p / std::cos(lat) - n;
    lat = std::atan2(e[2], p * (1.0 - kE2 * n / (n + h)));
  }
  return {lat / kDeg, lon / kDeg, h};
}

Vec3 to_enu(const Geodetic& g, const Geodetic& origin) {
  return mul(enu_basis(origin), sub(to_ecef(g), to_ecef(origin)));
}

Geodetic from_enu(const Vec3& enu, const Geodetic& origin) {
  const Mat3 r = enu_basis(origin);  // orthonormal: the inverse is the transpose
  const Vec3 o = to_ecef(origin);
  Vec3 e{};
  for (int j = 0; j < 3; ++j) e[j] = o[j] + r[0][j] * enu[0] + r[1][j] * enu[1] + r[2][j] * enu[2];
  return from_ecef(e);
}

Mat3 rotation_between(Vec3 from, Vec3 to) {
  from = normalized(from);
  to = normalized(to);
  const double c = dot(from, to);
  if (c < -1.0 + 1e-12) {
    // Opposite: rotate pi about any axis perpendicular to `from`.
    Vec3 axis = cross(from, std::abs(from[0]) < 0.9 ? Vec3{1, 0, 0} : Vec3{0, 1, 0});
    axis = normalized(axis);
    Mat3 r{};
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) r[i][j] = 2 * axis[i] * axis[j] - (i == j ? 1.0 : 0.0);
    return r;
  }
  // Rodrigues: R = I + [v]x + [v]x^2 / (1 + c), v = from x to
  const Vec3 v = cross(from, to);
  const Mat3 vx{{{0, -v[2], v[1]}, {v[2], 0, -v[0]}, {-v[1], v[0], 0}}};
  const Mat3 vx2 = mul(vx, vx);
  Mat3 r{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) r[i][j] = (i == j ? 1.0 : 0.0) + vx[i][j] + vx2[i][j] / (1.0 + c);
  return r;
}

Vec3 Georef::to_enu(const Vec3& scene) const {
  const Vec3 q = mul(rotation, scene);
  return {scale * q[0] + translation[0], scale * q[1] + translation[1], scale * q[2] + translation[2]};
}

Geodetic Georef::to_geodetic(const Vec3& scene) const { return from_enu(to_enu(scene), origin); }

Georef fit(const std::vector<PathSample>& path, const Vec3& up, const std::vector<GpsSample>& gps,
           const FitOptions& opts) {
  Georef g;
  const auto reject = [&](std::string why) {
    g.ok = false;
    g.reason = std::move(why);
    return g;
  };
  if (norm(up) < 1e-9) return reject("scene up vector is zero");

  // GPS fixes in time order, without "no fix" (0, 0) and without the repeats the car writes
  // between receiver updates.
  std::vector<GpsSample> sorted = gps;
  std::stable_sort(sorted.begin(), sorted.end(), [](const auto& x, const auto& y) { return x.t < y.t; });
  std::vector<GpsSample> fixes;
  for (const auto& s : sorted) {
    if (!std::isfinite(s.t) || !std::isfinite(s.lat_deg) || !std::isfinite(s.lon_deg)) continue;
    if (s.lat_deg == 0 && s.lon_deg == 0) continue;
    if (!fixes.empty() && fixes.back().lat_deg == s.lat_deg && fixes.back().lon_deg == s.lon_deg) continue;
    fixes.push_back(s);
  }
  if (fixes.size() < 2) return reject("fewer than two distinct GPS fixes in the window");

  g.origin = {fixes.front().lat_deg, fixes.front().lon_deg, 0};
  std::vector<Vec3> enu;
  enu.reserve(fixes.size());
  double e0 = 1e300, e1 = -1e300, n0 = 1e300, n1 = -1e300;
  for (const auto& f : fixes) {
    enu.push_back(to_enu({f.lat_deg, f.lon_deg, 0}, g.origin));
    e0 = std::min(e0, enu.back()[0]), e1 = std::max(e1, enu.back()[0]);
    n0 = std::min(n0, enu.back()[1]), n1 = std::max(n1, enu.back()[1]);
  }
  g.extent_m = std::hypot(e1 - e0, n1 - n0);
  if (g.extent_m < opts.min_extent_m) return reject("the car moved too little for a GPS fit");

  // Level the scene, then pair each path sample with the GPS position interpolated at its time.
  g.level = rotation_between(up, {0, 0, 1});
  std::vector<C> a, b;
  std::vector<double> heights;
  for (const auto& s : path) {
    // GPS position at s.t, linear between the fixes either side; no extrapolation, no bridging
    // of dropouts longer than max_gap_s.
    const auto hi = std::lower_bound(fixes.begin(), fixes.end(), s.t, [](const auto& f, double t) { return f.t < t; });
    if (hi == fixes.end()) continue;
    const auto j = static_cast<std::size_t>(hi - fixes.begin());
    Vec3 pos = enu[j];
    if (hi->t != s.t) {
      if (j == 0 || hi->t - fixes[j - 1].t > opts.max_gap_s) continue;
      const double w = (s.t - fixes[j - 1].t) / (hi->t - fixes[j - 1].t);
      for (int k = 0; k < 2; ++k) pos[k] = enu[j - 1][k] + w * (enu[j][k] - enu[j - 1][k]);
    }
    const Vec3 q = mul(g.level, s.p);
    a.emplace_back(q[0], q[1]);
    b.emplace_back(pos[0], pos[1]);
    heights.push_back(q[2]);
  }
  g.matched = a.size();
  if (g.matched < 5) return reject("fewer than five path samples overlap the GPS track");

  // RANSAC over 2-point samples. Small problems try every pair; big ones sample.
  const std::size_t n = a.size();
  const double thr2 = opts.inlier_m * opts.inlier_m;
  const auto inliers_of = [&](const Sim2& s) {
    std::vector<std::size_t> in;
    for (std::size_t i = 0; i < n; ++i)
      if (std::norm(s.apply(a[i]) - b[i]) <= thr2) in.push_back(i);
    return in;
  };
  const auto from_pair = [&](std::size_t i, std::size_t j) -> std::optional<Sim2> {
    const C da = a[j] - a[i];
    if (std::abs(b[j] - b[i]) < 2.0 * opts.inlier_m || std::norm(da) < 1e-18) return std::nullopt;  // ill-conditioned
    const C z = (b[j] - b[i]) / da;
    return Sim2{z, b[i] - z * a[i]};
  };
  std::vector<std::size_t> best;
  const auto consider = [&](std::size_t i, std::size_t j) {
    if (const auto s = from_pair(i, j)) {
      auto in = inliers_of(*s);
      if (in.size() > best.size()) best = std::move(in);
    }
  };
  if (n * (n - 1) / 2 <= static_cast<std::size_t>(opts.ransac_iters)) {
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = i + 1; j < n; ++j) consider(i, j);
  } else {
    std::mt19937_64 rng(opts.seed);
    std::uniform_int_distribution<std::size_t> pick(0, n - 1);
    for (int k = 0; k < opts.ransac_iters; ++k) {
      const std::size_t i = pick(rng), j = pick(rng);
      if (i != j) consider(i, j);
    }
  }
  if (best.size() < 2) return reject("no consistent GPS alignment (track too short or too noisy)");

  // Least squares on the inliers, re-selecting inliers until the set settles.
  std::optional<Sim2> sim;
  for (int round = 0; round < 5; ++round) {
    sim = fit_sim2(a, b, best);
    if (!sim) return reject("degenerate inlier set");
    auto in = inliers_of(*sim);
    if (in == best) break;
    if (in.size() < 2) return reject("inlier set collapsed during refinement");
    best = std::move(in);
  }
  g.inliers = best.size();
  if (g.inliers * 2 < g.matched) return reject("under half of the path agrees with the GPS track");

  double sum2 = 0;
  for (const auto i : best) {
    const double r = std::abs(sim->apply(a[i]) - b[i]);
    sum2 += r * r;
    g.max_m = std::max(g.max_m, r);
  }
  g.rms_m = std::sqrt(sum2 / static_cast<double>(best.size()));

  g.scale = std::abs(sim->z);
  g.yaw_rad = std::arg(sim->z);
  g.rotation = mul(rot_z(g.yaw_rad), g.level);
  // SEI has no altitude: put the path's mean height at ENU z = 0.
  double mean_h = 0;
  for (const auto h : heights) mean_h += h;
  mean_h /= static_cast<double>(heights.size());
  g.translation = {sim->t.real(), sim->t.imag(), -g.scale * mean_h};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) g.matrix[static_cast<std::size_t>(4 * i + j)] = g.scale * g.rotation[i][j];
    g.matrix[static_cast<std::size_t>(4 * i + 3)] = g.translation[static_cast<std::size_t>(i)];
  }
  g.matrix[15] = 1;
  g.ok = true;
  return g;
}

}  // namespace otrv::geo
