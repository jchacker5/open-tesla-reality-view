#include <doctest/doctest.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <numbers>
#include <random>
#include <string>
#include <vector>

#include "otrv/splat.hpp"

namespace splat = otrv::splat;
namespace fs = std::filesystem;

namespace {

constexpr double kC0 = 0.28209479177387814;

// One Gaussian with everything spelled out, in natural units.
struct G {
  splat::Vec3 p{0, 0, 5};
  splat::Vec3 scale{0.1, 0.1, 0.1};
  std::array<double, 4> q{1, 0, 0, 0};
  double opacity = 0.8;
  std::array<double, 3> rgb{0.9, 0.4, 0.1};  // degree-0 colour
};

splat::Gaussians make(const std::vector<G>& gs) {
  splat::Gaussians out;
  for (const auto& g : gs) {
    for (int a = 0; a < 3; ++a) {
      out.position.push_back(static_cast<float>(g.p[a]));
      out.log_scale.push_back(static_cast<float>(std::log(g.scale[a])));
      out.sh.push_back(static_cast<float>((g.rgb[a] - 0.5) / kC0));
    }
    for (const double v : g.q) out.rotation.push_back(static_cast<float>(v));
    out.opacity.push_back(static_cast<float>(std::log(g.opacity / (1 - g.opacity))));
  }
  return out;
}

// A camera at the origin looking down +z: focal 100 px, principal point at the image centre.
splat::Camera camera(int w = 65, int h = 65, double f = 100) {
  splat::Camera c;
  c.width = w, c.height = h;
  c.fx = c.fy = f;
  c.cx = 0.5 * w, c.cy = 0.5 * h;
  return c;
}

std::array<float, 3> px(const splat::Image& im, int x, int y) {
  const auto i = static_cast<std::size_t>(y * im.width + x) * 3;
  return {im.rgb[i], im.rgb[i + 1], im.rgb[i + 2]};
}

// Writes a PLY with properties in a deliberately scrambled order and mixed types.
void write_ply(const fs::path& path, const std::vector<std::map<std::string, double>>& verts,
               const std::vector<std::pair<std::string, std::string>>& props,
               const std::string& format = "binary_little_endian") {
  std::ofstream f(path, std::ios::binary);
  f << "ply\nformat " << format << " 1.0\ncomment test\nelement vertex " << verts.size() << "\n";
  for (const auto& [type, name] : props) f << "property " << type << " " << name << "\n";
  f << "end_header\n";
  for (const auto& v : verts)
    for (const auto& [type, name] : props) {
      const double x = v.count(name) ? v.at(name) : 0.0;
      if (type == "float") {
        const auto y = static_cast<float>(x);
        f.write(reinterpret_cast<const char*>(&y), 4);
      } else if (type == "double") {
        f.write(reinterpret_cast<const char*>(&x), 8);
      } else if (type == "uchar") {
        const auto y = static_cast<std::uint8_t>(x);
        f.write(reinterpret_cast<const char*>(&y), 1);
      }
    }
}

fs::path tmp(const std::string& name) { return fs::temp_directory_path() / ("otrv_test_" + name); }

}  // namespace

TEST_CASE("a single Gaussian matches the closed form") {
  // Mean at the centre of pixel (32, 32). Isotropic sigma 0.1 at depth 5 with f = 100 is 2 px, so the
  // 2D covariance is (2^2 + 0.3) I and alpha(r) = opacity * exp(-r^2 / (2 * 4.3)).
  auto cam = camera();
  cam.cx = cam.cy = 32.5;
  const G g;
  const auto im = splat::render(make({g}), cam);
  const double var = 4.0 + 0.3;
  for (const auto& [dx, dy] : {std::pair{0, 0}, {3, 0}, {0, -2}, {2, 2}, {5, 1}}) {
    const double a = g.opacity * std::exp(-0.5 * (dx * dx + dy * dy) / var);
    CAPTURE(dx);
    CAPTURE(dy);
    const auto c = px(im, 32 + dx, 32 + dy);
    if (a < 1.0 / 255) {
      CHECK(c[0] == 0.0f);
      continue;
    }
    for (int ch = 0; ch < 3; ++ch)
      CHECK(c[static_cast<std::size_t>(ch)] == doctest::Approx(g.rgb[static_cast<std::size_t>(ch)] * a).epsilon(1e-4));
    CHECK(im.alpha[static_cast<std::size_t>((32 + dy) * im.width + 32 + dx)] == doctest::Approx(a).epsilon(1e-4));
  }
}

TEST_CASE("background shows through by transmittance") {
  splat::RenderOptions o;
  o.background = {0.0f, 0.0f, 1.0f};
  auto cam = camera();
  cam.cx = cam.cy = 32.5;
  const auto im = splat::render(make({G{}}), cam, o);
  const auto c = px(im, 32, 32);
  CHECK(c[2] == doctest::Approx(0.1 * 0.8 + 0.2).epsilon(1e-4));  // 0.2 of the background left
  CHECK(px(im, 0, 0)[2] == 1.0f);
}

TEST_CASE("nearer Gaussians occlude farther ones whatever the file order") {
  G nearer, farther;
  nearer.p = {0, 0, 4}, nearer.rgb = {1, 0, 0}, nearer.opacity = 0.99, nearer.scale = {0.3, 0.3, 0.3};
  farther.p = {0, 0, 8}, farther.rgb = {0, 1, 0}, farther.opacity = 0.99, farther.scale = {0.6, 0.6, 0.6};
  for (const auto& order : {std::vector<G>{nearer, farther}, std::vector<G>{farther, nearer}}) {
    const auto c = px(splat::render(make(order), camera()), 32, 32);
    CHECK(c[0] > 0.95f);
    CHECK(c[1] < 0.05f);
  }
}

TEST_CASE("anisotropy and rotation") {
  G g;
  g.scale = {0.4, 0.02, 0.02};  // a needle along x
  const auto horizontal = splat::render(make({g}), camera());
  CHECK(px(horizontal, 32 + 6, 32)[0] > 0.3f);
  CHECK(px(horizontal, 32, 32 + 6)[0] == 0.0f);
  g.q = {std::cos(std::numbers::pi / 4), 0, 0, std::sin(std::numbers::pi / 4)};  // 90 degrees about z: now along y
  const auto vertical = splat::render(make({g}), camera());
  CHECK(px(vertical, 32 + 6, 32)[0] == 0.0f);
  CHECK(px(vertical, 32, 32 + 6)[0] > 0.3f);
}

TEST_CASE("culling: behind the camera and outside the image") {
  G behind;
  behind.p = {0, 0, -5};
  G off;
  off.p = {50, 0, 5};
  splat::RenderStats st;
  const auto im = splat::render(make({behind, off}), camera(), {}, &st);
  CHECK(st.visible == 0);
  for (const float v : im.rgb) CHECK(v == 0.0f);
}

TEST_CASE("spherical harmonics: degree 1 colour depends on the view direction") {
  // Coefficient 3 multiplies -C1 * x: positive for rays going left (x < 0).
  splat::Gaussians g = make({G{}});
  g.sh_degree = 1;
  std::vector<float> sh(4 * 3, 0.0f);
  for (int c = 0; c < 3; ++c) sh[static_cast<std::size_t>(c)] = g.sh[static_cast<std::size_t>(c)];
  sh[3 * 3 + 0] = 1.0f;  // red, coefficient 3
  g.sh = sh;
  g.position = {0, 0, 0};
  const auto from = [&](double x) {
    const auto cam = splat::Camera::look_at({x, 0, -5}, {0, 0, 0}, {0, -1, 0}, 40, 33, 33);
    return px(splat::render(g, cam), 16, 16)[0];
  };
  const double c1 = 0.4886025119029199, d = 3.0 / std::sqrt(9.0 + 25.0);
  // camera at +x: the ray to the Gaussian points to -x, so red rises by C1 * d (times alpha 0.8)
  CHECK(from(3) == doctest::Approx(0.8 * (0.9 + c1 * d)).epsilon(1e-3));
  CHECK(from(-3) == doctest::Approx(0.8 * (0.9 - c1 * d)).epsilon(1e-3));
}

TEST_CASE("look_at") {
  const auto cam = splat::Camera::look_at({1, 2, 3}, {4, 2, 7}, {0, 1, 0}, 60, 201, 101);  // odd: centre of pixel (100, 50)
  const auto c = cam.centre();
  CHECK(c[0] == doctest::Approx(1));
  CHECK(c[1] == doctest::Approx(2));
  CHECK(c[2] == doctest::Approx(3));
  // the target projects to the principal point, and world up is image up (-y)
  G g;
  g.p = {4, 2, 7};
  const auto im = splat::render(make({g}), cam);
  float best = 0;
  int bx = 0, by = 0;
  for (int y = 0; y < 101; ++y)
    for (int x = 0; x < 201; ++x)
      if (px(im, x, y)[0] > best) best = px(im, x, y)[0], bx = x, by = y;
  CHECK(bx == 100);
  CHECK(by == 50);
  g.p = {4, 3, 7};
  const auto im2 = splat::render(make({g}), cam);
  int by2 = 0;
  best = 0;
  for (int y = 0; y < 101; ++y)
    if (px(im2, 100, y)[0] > best) best = px(im2, 100, y)[0], by2 = y;
  CHECK(by2 < 50);
}

TEST_CASE("output does not depend on the thread count") {
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> u(-1, 1), s(0.01, 0.3), o(0.05, 0.95);
  std::vector<G> gs(3000);
  for (auto& g : gs) {
    g.p = {3 * u(rng), 2 * u(rng), 6 + 3 * u(rng)};
    g.scale = {s(rng), s(rng), s(rng)};
    g.q = {u(rng), u(rng), u(rng), u(rng)};
    g.opacity = o(rng);
    g.rgb = {o(rng), o(rng), o(rng)};
  }
  const auto scene = make(gs);
  const auto cam = camera(320, 200, 250);
  splat::RenderOptions one, many;
  one.threads = 1;
  many.threads = 8;
  const auto a = splat::render(scene, cam, one), b = splat::render(scene, cam, many);
  CHECK(std::memcmp(a.rgb.data(), b.rgb.data(), a.rgb.size() * sizeof(float)) == 0);
}

TEST_CASE("PLY loading") {
  const auto path = tmp("scene.ply");
  std::map<std::string, double> v{{"x", 1},        {"y", 2},        {"z", 3},        {"opacity", 0.5}, {"scale_0", -1},
                                  {"scale_1", -2}, {"scale_2", -3}, {"rot_0", 1},    {"rot_1", 0},     {"rot_2", 0},
                                  {"rot_3", 0},    {"f_dc_0", 0.1}, {"f_dc_1", 0.2}, {"f_dc_2", 0.3},  {"nx", 9}};
  for (int i = 0; i < 9; ++i) v["f_rest_" + std::to_string(i)] = 10 + i;  // degree 1
  // Scrambled order (Brush writes them alphabetically), a double and an unrelated uchar.
  const std::vector<std::pair<std::string, std::string>> props{
      {"float", "f_dc_0"},   {"float", "f_dc_1"},   {"float", "f_dc_2"},   {"float", "f_rest_0"}, {"float", "f_rest_1"},
      {"float", "f_rest_2"}, {"float", "f_rest_3"}, {"float", "f_rest_4"}, {"float", "f_rest_5"}, {"float", "f_rest_6"},
      {"float", "f_rest_7"}, {"float", "f_rest_8"}, {"uchar", "nx"},       {"float", "opacity"},  {"float", "rot_0"},
      {"float", "rot_1"},    {"float", "rot_2"},    {"float", "rot_3"},    {"float", "scale_0"},  {"float", "scale_1"},
      {"float", "scale_2"},  {"double", "x"},       {"float", "y"},        {"float", "z"}};
  write_ply(path, {v, v}, props);
  const auto g = splat::load_ply(path);
  REQUIRE(g.size() == 2);
  CHECK(g.sh_degree == 1);
  CHECK(g.position[3] == 1.0f);
  CHECK(g.log_scale[2] == -3.0f);
  CHECK(g.opacity[1] == 0.5f);
  // coefficient-major: sh[k * 3 + c]; f_rest is channel-major, so red's 3 rest coefficients are 10, 11, 12
  CHECK(g.sh[0] == doctest::Approx(0.1f));
  CHECK(g.sh[1 * 3 + 0] == 10.0f);
  CHECK(g.sh[2 * 3 + 0] == 11.0f);
  CHECK(g.sh[3 * 3 + 0] == 12.0f);
  CHECK(g.sh[1 * 3 + 1] == 13.0f);
  CHECK(g.sh[1 * 3 + 2] == 16.0f);

  SUBCASE("errors") {
    write_ply(path, {v}, props, "ascii");
    CHECK_THROWS_AS((void)splat::load_ply(path), splat::PlyError);
    write_ply(path, {v}, {{"float", "x"}, {"float", "y"}, {"float", "z"}});
    CHECK_THROWS_WITH_AS((void)splat::load_ply(path), doctest::Contains("has no 'scale_0'"), splat::PlyError);
    write_ply(path, {v, v}, props);
    fs::resize_file(path, fs::file_size(path) - 10);
    CHECK_THROWS_WITH_AS((void)splat::load_ply(path), doctest::Contains("truncated"), splat::PlyError);
    CHECK_THROWS_AS((void)splat::load_ply(tmp("missing.ply")), splat::PlyError);
  }
  fs::remove(path);
}

TEST_CASE("PNG output is a valid image") {
  const auto path = tmp("out.png");
  std::vector<std::uint8_t> rgb(70000 * 3);  // more than one 64 KiB stored block
  for (std::size_t i = 0; i < rgb.size(); ++i) rgb[i] = static_cast<std::uint8_t>(i * 7);
  splat::write_png(path, 350, 200, rgb);
  std::ifstream f(path, std::ios::binary);
  std::vector<unsigned char> head(16);
  f.read(reinterpret_cast<char*>(head.data()), 16);
  CHECK(head[0] == 0x89);
  CHECK(std::string(head.begin() + 1, head.begin() + 4) == "PNG");
  CHECK(std::string(head.begin() + 12, head.begin() + 16) == "IHDR");
  fs::remove(path);
  CHECK_THROWS((void)splat::write_png(path, 10, 10, rgb));
}
