// A CPU renderer for 3D Gaussian splats.
//
// The forward pass of 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023), the same one Brush and
// the browser viewer run on the GPU: project each Gaussian's 3D covariance to a 2D ellipse through
// the local affine approximation of the perspective projection, bin the ellipses into 16x16 pixel
// tiles, sort each tile's list by depth, and alpha-composite front to back with early termination.
// Colour is degree 0-3 spherical harmonics evaluated along the viewing ray.
//
// It runs headless on any machine (no GPU, no browser), renders a frame deterministically for a
// given scene and camera whatever the thread count, and is checked against Brush's own renders of
// the same views. Uses: snapshots and thumbnails from the CLI, CI, and a reference for the GPU paths.
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace otrv::splat {

using Vec3 = std::array<double, 3>;
using Mat3 = std::array<std::array<double, 3>, 3>;

// Gaussians in structure-of-arrays form, as stored in a 3DGS / Brush PLY.
struct Gaussians {
  std::size_t size() const { return opacity.size(); }

  std::vector<float> position;   // 3 per Gaussian
  std::vector<float> log_scale;  // 3 per Gaussian (the file stores log axis lengths)
  std::vector<float> rotation;   // 4 per Gaussian, quaternion w, x, y, z (not necessarily normalised)
  std::vector<float> opacity;    // logit
  int sh_degree = 0;             // 0..3
  // (sh_degree + 1)^2 RGB coefficients per Gaussian, coefficient-major: sh[(i * K + k) * 3 + c]
  std::vector<float> sh;
};

class PlyError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Reads a binary little-endian PLY with a `vertex` element holding x y z, opacity, scale_0..2,
// rot_0..3, f_dc_0..2 and optionally f_rest_* (3DGS layout: channel-major, 3 * (K - 1) of them).
// Properties may come in any order and any scalar type; unknown ones are ignored.
[[nodiscard]] Gaussians load_ply(const std::filesystem::path& path);

// A pinhole camera in COLMAP's convention: x right, y down, z forward; world -> camera is
// p_cam = R p_world + t; pixel centres are at integer + 0.5.
struct Camera {
  int width = 0, height = 0;
  double fx = 0, fy = 0, cx = 0, cy = 0;
  Mat3 R{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
  Vec3 t{0, 0, 0};

  [[nodiscard]] Vec3 centre() const;  // camera position in world space

  // A camera at `eye` looking at `target`, with `up` (world space) pointing up in the image.
  [[nodiscard]] static Camera look_at(const Vec3& eye, const Vec3& target, const Vec3& up, double vertical_fov_deg,
                                      int width, int height);
  // From a COLMAP quaternion (w, x, y, z) and translation, as in images.txt.
  [[nodiscard]] static Camera from_colmap(const std::array<double, 4>& qvec, const Vec3& tvec, int width, int height,
                                          double fx, double fy, double cx, double cy);
};

struct RenderOptions {
  std::array<float, 3> background{0, 0, 0};
  int threads = 0;           // 0 = all hardware threads
  double near_plane = 0.01;  // camera-space z below which a Gaussian is culled
  float dilation = 0.3f;     // added to the 2D covariance diagonal (anti-aliasing low-pass, as in 3DGS)
};

struct Image {
  int width = 0, height = 0;
  std::vector<float> rgb;    // linear-in-file colour (the splat's SH space), row-major, 3 per pixel
  std::vector<float> alpha;  // accumulated opacity per pixel

  [[nodiscard]] std::vector<std::uint8_t> to_rgb8() const;  // clamped to [0, 1], rounded
};

struct RenderStats {
  std::size_t visible = 0;    // Gaussians that survived culling
  std::size_t tile_hits = 0;  // (Gaussian, tile) pairs composited
  double project_ms = 0, sort_ms = 0, raster_ms = 0;
};

[[nodiscard]] Image render(const Gaussians& g, const Camera& cam, const RenderOptions& opts = {},
                           RenderStats* stats = nullptr);

// Writes an 8-bit RGB PNG (uncompressed deflate, so no zlib dependency).
void write_png(const std::filesystem::path& path, int width, int height, const std::vector<std::uint8_t>& rgb);

}  // namespace otrv::splat
