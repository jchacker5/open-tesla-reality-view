#include "otrv/splat.hpp"
#include "otrv/detail/parallel.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <numbers>
#include <sstream>
#include <thread>

namespace otrv::splat {
namespace {

constexpr int kTile = 16;
constexpr float kMinAlpha = 1.0f / 255.0f;
constexpr float kMaxAlpha = 0.99f;
constexpr float kMinTransmittance = 1e-4f;

// ---------------------------------------------------------------------------------------------
// PLY

struct PlyProperty {
  std::string type;
  std::size_t offset = 0;
};

std::size_t type_size(const std::string& t) {
  if (t == "char" || t == "int8" || t == "uchar" || t == "uint8") return 1;
  if (t == "short" || t == "int16" || t == "ushort" || t == "uint16") return 2;
  if (t == "int" || t == "int32" || t == "uint" || t == "uint32" || t == "float" || t == "float32") return 4;
  if (t == "double" || t == "float64") return 8;
  throw PlyError("unsupported PLY property type '" + t + "'");
}

template <typename T>
T load_le(const std::uint8_t* p) {
  std::array<std::uint8_t, sizeof(T)> b;
  std::memcpy(b.data(), p, sizeof(T));
  if constexpr (std::endian::native == std::endian::big) std::reverse(b.begin(), b.end());
  return std::bit_cast<T>(b);
}

float read_scalar(const std::uint8_t* p, const std::string& t) {
  if (t == "float" || t == "float32") return load_le<float>(p);
  if (t == "double" || t == "float64") return static_cast<float>(load_le<double>(p));
  if (t == "uchar" || t == "uint8") return p[0];
  if (t == "char" || t == "int8") return static_cast<std::int8_t>(p[0]);
  if (t == "ushort" || t == "uint16") return load_le<std::uint16_t>(p);
  if (t == "short" || t == "int16") return load_le<std::int16_t>(p);
  if (t == "uint" || t == "uint32") return static_cast<float>(load_le<std::uint32_t>(p));
  return static_cast<float>(load_le<std::int32_t>(p));
}

// ---------------------------------------------------------------------------------------------
// Maths

using Mat2 = std::array<double, 3>;  // symmetric 2x2: a, b, c = [[a, b], [b, c]]

Mat3 quat_to_matrix(double w, double x, double y, double z) {
  const double n = std::sqrt(w * w + x * x + y * y + z * z);
  if (n == 0) return {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
  w /= n, x /= n, y /= n, z /= n;
  return {{{1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)},
           {2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)},
           {2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)}}};
}

Vec3 mul(const Mat3& m, const Vec3& v) {
  return {m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2], m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
          m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2]};
}

Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

Vec3 normalized(const Vec3& v) {
  const double n = std::hypot(v[0], v[1], v[2]);
  if (!std::isfinite(n) || n <= 0) throw std::invalid_argument("camera orientation must be finite and nondegenerate");
  return {v[0] / n, v[1] / n, v[2] / n};
}

// Real spherical harmonics up to degree 3, the 3DGS basis and sign convention.
constexpr double kC0 = 0.28209479177387814;
constexpr double kC1 = 0.4886025119029199;
constexpr std::array<double, 5> kC2{1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792,
                                    0.5462742152960396};
constexpr std::array<double, 7> kC3{-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
                                    -0.4570457994644658, 1.445305721320277, -0.5900435899266435};

std::array<float, 3> eval_sh(int degree, const float* sh, const Vec3& d) {
  std::array<double, 16> basis{};
  basis[0] = kC0;
  if (degree > 0) {
    const double x = d[0], y = d[1], z = d[2];
    basis[1] = -kC1 * y;
    basis[2] = kC1 * z;
    basis[3] = -kC1 * x;
    if (degree > 1) {
      const double xx = x * x, yy = y * y, zz = z * z, xy = x * y, yz = y * z, xz = x * z;
      basis[4] = kC2[0] * xy;
      basis[5] = kC2[1] * yz;
      basis[6] = kC2[2] * (2 * zz - xx - yy);
      basis[7] = kC2[3] * xz;
      basis[8] = kC2[4] * (xx - yy);
      if (degree > 2) {
        basis[9] = kC3[0] * y * (3 * xx - yy);
        basis[10] = kC3[1] * xy * z;
        basis[11] = kC3[2] * y * (4 * zz - xx - yy);
        basis[12] = kC3[3] * z * (2 * zz - 3 * xx - 3 * yy);
        basis[13] = kC3[4] * x * (4 * zz - xx - yy);
        basis[14] = kC3[5] * z * (xx - yy);
        basis[15] = kC3[6] * x * (xx - 3 * yy);
      }
    }
  }
  const int k = (degree + 1) * (degree + 1);
  std::array<float, 3> rgb{};
  for (int c = 0; c < 3; ++c) {
    double v = 0.5;
    for (int i = 0; i < k; ++i) v += basis[static_cast<std::size_t>(i)] * sh[i * 3 + c];
    rgb[static_cast<std::size_t>(c)] =
        static_cast<float>(std::clamp(v, 0.0, static_cast<double>(std::numeric_limits<float>::max())));
  }
  return rgb;
}

// A Gaussian projected to the image.
struct Splat2D {
  float mx, my;      // mean, pixels
  float ca, cb, cc;  // conic: inverse 2D covariance [[ca, cb], [cb, cc]]
  float opacity;     // 0..1
  std::array<float, 3> rgb;
};

}  // namespace

// -----------------------------------------------------------------------------------------------

Gaussians load_ply(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw PlyError("cannot open " + path.string());

  std::string line;
  if (!std::getline(in, line) || (line != "ply" && line != "ply\r")) throw PlyError(path.string() + " is not a PLY file");
  std::size_t count = 0, stride = 0;
  std::map<std::string, PlyProperty> props;
  bool in_vertex = false, seen_vertex = false, format_ok = false, ended = false;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    std::istringstream ss(line);
    std::string word;
    ss >> word;
    if (word == "format") {
      std::string fmt, version;
      ss >> fmt >> version;
      if (fmt != "binary_little_endian") throw PlyError("only binary_little_endian PLY is supported, not " + fmt);
      if (format_ok || version != "1.0") throw PlyError("invalid PLY format declaration");
      format_ok = true;
    } else if (word == "element") {
      std::string name;
      ss >> name;
      in_vertex = name == "vertex";
      if (in_vertex) {
        if (seen_vertex) throw PlyError("duplicate PLY vertex element");
        std::string token;
        ss >> token;
        const auto [end, ec] = std::from_chars(token.data(), token.data() + token.size(), count);
        if (ec != std::errc{} || end != token.data() + token.size()) throw PlyError("invalid PLY vertex count");
        seen_vertex = true;
      } else if (!seen_vertex) {
        throw PlyError("elements before 'vertex' are not supported");
      }
    } else if (word == "property" && in_vertex) {
      std::string type, name;
      ss >> type;
      if (type == "list") throw PlyError("list properties in 'vertex' are not supported");
      ss >> name;
      if (name.empty() || props.count(name)) throw PlyError("missing or duplicate PLY property name");
      const auto width = type_size(type);
      if (stride > std::numeric_limits<std::size_t>::max() - width) throw PlyError("PLY vertex stride is too large");
      props.emplace(name, PlyProperty{type, stride});
      stride += width;
    } else if (word == "end_header") {
      ended = true;
      break;
    }
  }
  if (!format_ok) throw PlyError("PLY header has no format line");
  if (!seen_vertex) throw PlyError("PLY has no vertex element");
  if (!ended) throw PlyError("PLY has no end_header");

  const auto need = [&](const std::string& n) -> const PlyProperty& {
    const auto it = props.find(n);
    if (it == props.end()) throw PlyError("PLY vertex has no '" + n + "' property (not a Gaussian splat?)");
    return it->second;
  };
  const std::array<const PlyProperty*, 3> pos{&need("x"), &need("y"), &need("z")};
  const std::array<const PlyProperty*, 3> scl{&need("scale_0"), &need("scale_1"), &need("scale_2")};
  const std::array<const PlyProperty*, 4> rot{&need("rot_0"), &need("rot_1"), &need("rot_2"), &need("rot_3")};
  const std::array<const PlyProperty*, 3> dc{&need("f_dc_0"), &need("f_dc_1"), &need("f_dc_2")};
  const PlyProperty& op = need("opacity");
  std::size_t rest = 0;
  while (props.count("f_rest_" + std::to_string(rest))) ++rest;
  const auto rest_total = static_cast<std::size_t>(std::count_if(props.begin(), props.end(), [](const auto& p) {
    return p.first.starts_with("f_rest_");
  }));
  if (rest != rest_total || rest > 45) throw PlyError("invalid f_rest_* property sequence");
  int degree = 0;
  while ((degree + 1) * (degree + 1) - 1 < static_cast<int>(rest / 3) && degree < 3) ++degree;
  const auto k = static_cast<std::size_t>((degree + 1) * (degree + 1));
  if (rest != 3 * (k - 1)) throw PlyError("unexpected number of f_rest_* properties: " + std::to_string(rest));
  std::vector<const PlyProperty*> rest_props;
  for (std::size_t i = 0; i < rest; ++i) rest_props.push_back(&props["f_rest_" + std::to_string(i)]);

  // Validate against actual payload bytes before multiplying counts or allocating any buffers.
  const auto begin = in.tellg();
  in.seekg(0, std::ios::end);
  const auto end = in.tellg();
  if (begin < 0 || end < begin) throw PlyError("cannot determine PLY payload size");
  const auto remaining = static_cast<std::uintmax_t>(end - begin);
  if (stride == 0 || count > remaining / stride) throw PlyError("PLY is truncated: vertex count exceeds payload");
  if (count > std::vector<float>().max_size() / (k * 3) ||
      count > std::vector<float>().max_size() / 4 ||
      count > static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max()) / stride)
    throw PlyError("PLY vertex count is too large");
  in.seekg(begin);
  std::vector<std::uint8_t> data(count * stride);
  if (!in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size())))
    throw PlyError("PLY is truncated: expected " + std::to_string(count) + " vertices");

  Gaussians g;
  g.sh_degree = degree;
  g.position.resize(3 * count);
  g.log_scale.resize(3 * count);
  g.rotation.resize(4 * count);
  g.opacity.resize(count);
  g.sh.resize(count * k * 3);
  for (std::size_t i = 0; i < count; ++i) {
    const std::uint8_t* rec = data.data() + i * stride;
    const auto get = [&](const PlyProperty* p) {
      const float value = read_scalar(rec + p->offset, p->type);
      if (!std::isfinite(value)) throw PlyError("nonfinite Gaussian property in PLY");
      return value;
    };
    for (std::size_t a = 0; a < 3; ++a) {
      g.position[3 * i + a] = get(pos[a]);
      g.log_scale[3 * i + a] = get(scl[a]);
      g.sh[(i * k) * 3 + a] = get(dc[a]);
    }
    for (std::size_t a = 0; a < 4; ++a) g.rotation[4 * i + a] = get(rot[a]);
    g.opacity[i] = get(&op);
    // f_rest is channel-major in the file: f_rest_{c * (k - 1) + j} is channel c, coefficient j + 1.
    for (std::size_t c = 0; c < 3; ++c)
      for (std::size_t j = 0; j + 1 < k; ++j) g.sh[(i * k + j + 1) * 3 + c] = get(rest_props[c * (k - 1) + j]);
  }
  return g;
}

Vec3 Camera::centre() const {
  // C = -R^T t
  return {-(R[0][0] * t[0] + R[1][0] * t[1] + R[2][0] * t[2]), -(R[0][1] * t[0] + R[1][1] * t[1] + R[2][1] * t[2]),
          -(R[0][2] * t[0] + R[1][2] * t[1] + R[2][2] * t[2])};
}

Camera Camera::look_at(const Vec3& eye, const Vec3& target, const Vec3& up, double vertical_fov_deg, int width,
                       int height) {
  if (!std::isfinite(vertical_fov_deg) || vertical_fov_deg <= 0 || vertical_fov_deg >= 180 || width <= 0 || height <= 0)
    throw std::invalid_argument("camera requires positive dimensions and a field of view between 0 and 180 degrees");
  const Vec3 z = normalized({target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]});  // forward
  const Vec3 x = normalized(cross(z, up));  // right (image x); up points to -y in the image
  const Vec3 y = cross(z, x);               // down
  Camera c;
  c.width = width;
  c.height = height;
  c.fy = 0.5 * height / std::tan(0.5 * vertical_fov_deg * std::numbers::pi / 180.0);
  c.fx = c.fy;
  c.cx = 0.5 * width;
  c.cy = 0.5 * height;
  c.R = {x, y, z};
  const Vec3 re = mul(c.R, eye);
  c.t = {-re[0], -re[1], -re[2]};
  return c;
}

Camera Camera::from_colmap(const std::array<double, 4>& q, const Vec3& tvec, int width, int height, double fx,
                           double fy, double cx, double cy) {
  Camera c;
  c.width = width;
  c.height = height;
  c.fx = fx, c.fy = fy, c.cx = cx, c.cy = cy;
  c.R = quat_to_matrix(q[0], q[1], q[2], q[3]);
  c.t = tvec;
  return c;
}

std::vector<std::uint8_t> Image::to_rgb8() const {
  std::vector<std::uint8_t> out(rgb.size());
  for (std::size_t i = 0; i < rgb.size(); ++i) {
    if (!std::isfinite(rgb[i])) throw std::invalid_argument("image contains nonfinite colour");
    out[i] = static_cast<std::uint8_t>(std::lround(std::clamp(rgb[i], 0.0f, 1.0f) * 255.0f));
  }
  return out;
}

Image render(const Gaussians& g, const Camera& cam, const RenderOptions& opts, RenderStats* stats) {
  using clock = std::chrono::steady_clock;
  const auto ms_since = [](clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
  };
  if (cam.width <= 0 || cam.height <= 0 || cam.width > std::numeric_limits<int>::max() - kTile ||
      cam.height > std::numeric_limits<int>::max() - kTile)
    throw std::invalid_argument("camera dimensions are outside the supported range");
  const auto all_finite = [](const auto& values) {
    return std::all_of(values.begin(), values.end(), [](auto value) { return std::isfinite(value); });
  };
  if (!(cam.fx > 0 && cam.fy > 0) || !all_finite(std::array{cam.fx, cam.fy, cam.cx, cam.cy}) ||
      !all_finite(cam.t) || !std::all_of(cam.R.begin(), cam.R.end(), all_finite))
    throw std::invalid_argument("camera intrinsics and pose must be finite, with positive focal lengths");
  if (!std::isfinite(opts.near_plane) || opts.near_plane <= 0 || !std::isfinite(opts.dilation) ||
      opts.dilation < 0 || opts.threads < 0 || !all_finite(opts.background))
    throw std::invalid_argument("invalid render options");
  const std::size_t n = g.size();
  if (g.sh_degree < 0 || g.sh_degree > 3 || n > std::numeric_limits<std::uint32_t>::max() ||
      n > std::vector<float>().max_size() / 48)
    throw std::invalid_argument("unsupported Gaussian count or SH degree");
  const auto k_sh = static_cast<std::size_t>((g.sh_degree + 1) * (g.sh_degree + 1));
  if (g.position.size() != 3 * n || g.log_scale.size() != 3 * n || g.rotation.size() != 4 * n ||
      g.sh.size() != n * k_sh * 3)
    throw std::invalid_argument("Gaussian arrays have inconsistent sizes");
  if (!all_finite(g.position) || !all_finite(g.log_scale) || !all_finite(g.rotation) ||
      !all_finite(g.opacity) || !all_finite(g.sh))
    throw std::invalid_argument("Gaussian arrays contain nonfinite values");
  const int tiles_x = 1 + (cam.width - 1) / kTile, tiles_y = 1 + (cam.height - 1) / kTile;
  const auto ntiles = static_cast<std::size_t>(tiles_x) * static_cast<std::size_t>(tiles_y);
  if (ntiles > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      static_cast<std::size_t>(cam.width) > std::vector<float>().max_size() / 3 / static_cast<std::size_t>(cam.height))
    throw std::invalid_argument("camera dimensions are too large");
  const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
  const unsigned nthreads = opts.threads > 0 ? static_cast<unsigned>(opts.threads) : hw;
  const auto parallel_for = [&](std::size_t count, auto&& body) {
    otrv::detail::parallel_for(count, nthreads, body);
  };

  // 1. Project every Gaussian (in parallel chunks).
  auto t0 = clock::now();
  std::vector<Splat2D> splats(n);
  std::vector<float> depth(n, -1.0f);       // < 0: culled
  std::vector<std::array<int, 4>> rect(n);  // tile rect [x0, y0, x1, y1)
  const Vec3 centre = cam.centre();
  const double lim_x = 1.3 * 0.5 * cam.width / cam.fx, lim_y = 1.3 * 0.5 * cam.height / cam.fy;
  constexpr std::size_t kChunk = 4096;
  parallel_for((n + kChunk - 1) / kChunk, [&](std::size_t chunk) {
    for (std::size_t i = chunk * kChunk; i < std::min(n, (chunk + 1) * kChunk); ++i) {
      const Vec3 pw{g.position[3 * i], g.position[3 * i + 1], g.position[3 * i + 2]};
      const Vec3 pc = mul(cam.R, pw);
      const double x = pc[0] + cam.t[0], y = pc[1] + cam.t[1], z = pc[2] + cam.t[2];
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || z < opts.near_plane ||
          z > std::numeric_limits<float>::max()) continue;

      // 3D covariance: Sigma = (R S)(R S)^T in world space, then into camera space with W = cam.R.
      const Mat3 rq =
          quat_to_matrix(g.rotation[4 * i], g.rotation[4 * i + 1], g.rotation[4 * i + 2], g.rotation[4 * i + 3]);
      const double s[3] = {std::exp(double{g.log_scale[3 * i]}), std::exp(double{g.log_scale[3 * i + 1]}),
                           std::exp(double{g.log_scale[3 * i + 2]})};
      Mat3 m{};  // T = W * R * S
      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) {
          double v = 0;
          for (int k = 0; k < 3; ++k) v += cam.R[r][k] * rq[k][c];
          m[r][c] = v * s[c];
        }
      // Jacobian of the projection at the (frustum-clamped) mean.
      const double tx = std::clamp(x / z, -lim_x, lim_x) * z, ty = std::clamp(y / z, -lim_y, lim_y) * z;
      const double j00 = cam.fx / z, j02 = -cam.fx * tx / (z * z), j11 = cam.fy / z, j12 = -cam.fy * ty / (z * z);
      // rows of J * T (2x3)
      double u[3], v[3];
      for (int c = 0; c < 3; ++c) {
        u[c] = j00 * m[0][c] + j02 * m[2][c];
        v[c] = j11 * m[1][c] + j12 * m[2][c];
      }
      Mat2 cov{u[0] * u[0] + u[1] * u[1] + u[2] * u[2] + opts.dilation, u[0] * v[0] + u[1] * v[1] + u[2] * v[2],
               v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + opts.dilation};
      const double det = cov[0] * cov[2] - cov[1] * cov[1];
      if (!std::isfinite(det) || det <= 0) continue;
      const double mid = 0.5 * (cov[0] + cov[2]);
      const double lambda = mid + std::sqrt(std::max(0.1, mid * mid - det));
      const double radius = std::ceil(3.0 * std::sqrt(lambda));
      const double mx = cam.fx * x / z + cam.cx, my = cam.fy * y / z + cam.cy;
      if (!std::isfinite(radius) || !std::isfinite(mx) || !std::isfinite(my) ||
          std::abs(mx) > std::numeric_limits<float>::max() || std::abs(my) > std::numeric_limits<float>::max()) continue;
      const auto tile_bound = [](double value, int limit) {
        return static_cast<int>(std::clamp(std::floor(value / kTile), 0.0, static_cast<double>(limit)));
      };
      const int x0 = tile_bound(mx - radius, tiles_x), x1 = tile_bound(mx + radius + kTile - 1, tiles_x);
      const int y0 = tile_bound(my - radius, tiles_y), y1 = tile_bound(my + radius + kTile - 1, tiles_y);
      if (x0 >= x1 || y0 >= y1) continue;

      const Vec3 dir = normalized({pw[0] - centre[0], pw[1] - centre[1], pw[2] - centre[2]});
      splats[i] = {static_cast<float>(mx),
                   static_cast<float>(my),
                   static_cast<float>(cov[2] / det),
                   static_cast<float>(-cov[1] / det),
                   static_cast<float>(cov[0] / det),
                   static_cast<float>(1.0 / (1.0 + std::exp(-double{g.opacity[i]}))),
                   eval_sh(g.sh_degree, &g.sh[i * k_sh * 3], dir)};
      depth[i] = static_cast<float>(z);
      rect[i] = {x0, y0, x1, y1};
    }
  });
  if (stats) stats->project_ms = ms_since(t0);

  // 2. Bin into tiles (counting sort), then sort each tile's list by (depth, index) in parallel.
  //    Same order as one global sort on (tile, depth, index), without its serial n log n.
  t0 = clock::now();
  std::vector<std::size_t> start(ntiles + 1, 0);
  std::size_t visible = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (depth[i] < 0) continue;
    ++visible;
    for (int ty = rect[i][1]; ty < rect[i][3]; ++ty)
      for (int tx = rect[i][0]; tx < rect[i][2]; ++tx) ++start[static_cast<std::size_t>(ty * tiles_x + tx) + 1];
  }
  for (std::size_t t = 0; t < ntiles; ++t) start[t + 1] += start[t];
  std::vector<std::uint32_t> order(start[ntiles]);
  {
    std::vector<std::size_t> fill(start.begin(), start.end() - 1);
    for (std::size_t i = 0; i < n; ++i) {
      if (depth[i] < 0) continue;
      for (int ty = rect[i][1]; ty < rect[i][3]; ++ty)
        for (int tx = rect[i][0]; tx < rect[i][2]; ++tx)
          order[fill[static_cast<std::size_t>(ty * tiles_x + tx)]++] = static_cast<std::uint32_t>(i);
    }
  }
  // Indices arrive in increasing order per tile, so a stable sort on depth breaks ties by index.
  parallel_for(ntiles, [&](std::size_t tile) {
    std::stable_sort(order.begin() + static_cast<std::ptrdiff_t>(start[tile]),
                     order.begin() + static_cast<std::ptrdiff_t>(start[tile + 1]),
                     [&](std::uint32_t a, std::uint32_t b) { return depth[a] < depth[b]; });
  });
  if (stats) {
    stats->sort_ms = ms_since(t0);
    stats->visible = visible;
    stats->tile_hits = order.size();
  }

  // 3. Composite front to back, one tile per task.
  t0 = clock::now();
  Image img;
  img.width = cam.width;
  img.height = cam.height;
  img.rgb.assign(static_cast<std::size_t>(cam.width) * cam.height * 3, 0.0f);
  img.alpha.assign(static_cast<std::size_t>(cam.width) * cam.height, 0.0f);
  parallel_for(ntiles, [&](std::size_t tile) {
    const int tx = static_cast<int>(tile) % tiles_x, ty = static_cast<int>(tile) / tiles_x;
    for (int py = ty * kTile; py < std::min(cam.height, (ty + 1) * kTile); ++py) {
      for (int px = tx * kTile; px < std::min(cam.width, (tx + 1) * kTile); ++px) {
        const float fx = static_cast<float>(px) + 0.5f, fy = static_cast<float>(py) + 0.5f;
        float T = 1.0f;
        std::array<float, 3> c{0, 0, 0};
        for (std::size_t k = start[tile]; k < start[tile + 1]; ++k) {
          const Splat2D& s = splats[order[k]];
          const double dx = static_cast<double>(s.mx) - fx, dy = static_cast<double>(s.my) - fy;
          const double power = -0.5 * (s.ca * dx * dx + s.cc * dy * dy) - s.cb * dx * dy;
          if (!std::isfinite(power) || power > 0) continue;
          const float a = static_cast<float>(std::min(static_cast<double>(kMaxAlpha), s.opacity * std::exp(power)));
          if (a < kMinAlpha) continue;
          const float next = T * (1 - a);
          if (next < kMinTransmittance) break;
          for (int ch = 0; ch < 3; ++ch) c[static_cast<std::size_t>(ch)] += s.rgb[static_cast<std::size_t>(ch)] * a * T;
          T = next;
        }
        const auto p =
            static_cast<std::size_t>(py) * static_cast<std::size_t>(cam.width) + static_cast<std::size_t>(px);
        for (std::size_t ch = 0; ch < 3; ++ch) img.rgb[3 * p + ch] = c[ch] + T * opts.background[ch];
        img.alpha[p] = 1 - T;
      }
    }
  });
  if (stats) stats->raster_ms = ms_since(t0);
  return img;
}

// -----------------------------------------------------------------------------------------------
// PNG

namespace {

std::uint32_t crc32(const std::uint8_t* p, std::size_t n, std::uint32_t crc = 0) {
  static const auto table = [] {
    std::array<std::uint32_t, 256> t{};
    for (std::uint32_t i = 0; i < 256; ++i) {
      std::uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      t[i] = c;
    }
    return t;
  }();
  crc = ~crc;
  for (std::size_t i = 0; i < n; ++i) crc = table[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
  return ~crc;
}

void put32(std::vector<std::uint8_t>& out, std::uint32_t v) {
  for (int s = 24; s >= 0; s -= 8) out.push_back(static_cast<std::uint8_t>(v >> s));
}

void chunk(std::ofstream& f, const char* type, const std::vector<std::uint8_t>& data) {
  if (data.size() > 0x7FFFFFFFu || data.size() > std::vector<std::uint8_t>().max_size() - 12)
    throw std::invalid_argument("PNG chunk is too large");
  std::vector<std::uint8_t> buf;
  buf.reserve(data.size() + 12);
  put32(buf, static_cast<std::uint32_t>(data.size()));
  buf.insert(buf.end(), type, type + 4);
  if (!data.empty()) buf.insert(buf.end(), data.begin(), data.end());
  put32(buf, crc32(buf.data() + 4, buf.size() - 4));
  f.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
}

}  // namespace

void write_png(const std::filesystem::path& path, int width, int height, const std::vector<std::uint8_t>& rgb) {
  if (width <= 0 || height <= 0 || rgb.size() != static_cast<std::size_t>(width) * height * 3)
    throw std::invalid_argument("write_png: size mismatch");
  // Raw scanlines, each prefixed by filter type 0.
  const auto row = static_cast<std::size_t>(width) * 3;
  std::vector<std::uint8_t> raw;
  raw.reserve((row + 1) * static_cast<std::size_t>(height));
  for (int y = 0; y < height; ++y) {
    raw.push_back(0);
    raw.insert(raw.end(), rgb.begin() + static_cast<std::ptrdiff_t>(y * row),
               rgb.begin() + static_cast<std::ptrdiff_t>((y + 1) * row));
  }
  // zlib stream of stored (uncompressed) deflate blocks.
  std::vector<std::uint8_t> z{0x78, 0x01};
  std::uint32_t s1 = 1, s2 = 0;  // Adler-32
  for (std::size_t pos = 0; pos < raw.size() || pos == 0;) {
    const std::size_t len = std::min<std::size_t>(65535, raw.size() - pos);
    const bool last = pos + len == raw.size();
    z.push_back(last ? 1 : 0);
    z.push_back(static_cast<std::uint8_t>(len)), z.push_back(static_cast<std::uint8_t>(len >> 8));
    z.push_back(static_cast<std::uint8_t>(~len)), z.push_back(static_cast<std::uint8_t>(~len >> 8));
    z.insert(z.end(), raw.begin() + static_cast<std::ptrdiff_t>(pos),
             raw.begin() + static_cast<std::ptrdiff_t>(pos + len));
    for (std::size_t i = pos; i < pos + len; ++i) s1 = (s1 + raw[i]) % 65521, s2 = (s2 + s1) % 65521;
    pos += len;
    if (last) break;
  }
  put32(z, (s2 << 16) | s1);

  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + path.string());
  static constexpr std::uint8_t kSig[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
  f.write(reinterpret_cast<const char*>(kSig), 8);
  std::vector<std::uint8_t> ihdr;
  put32(ihdr, static_cast<std::uint32_t>(width));
  put32(ihdr, static_cast<std::uint32_t>(height));
  ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0});  // 8-bit, RGB, deflate, adaptive filtering, no interlace
  chunk(f, "IHDR", ihdr);
  chunk(f, "IDAT", z);
  chunk(f, "IEND", {});
  if (!f) throw std::runtime_error("cannot write " + path.string());
}

}  // namespace otrv::splat
