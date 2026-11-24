// otrv-render: render a Gaussian-splat PLY to a PNG on the CPU.
//
//   otrv-render scene.ply --colmap work/drive1/dense/sparse --image front/00036.jpg -o view.png
//   otrv-render scene.ply --eye 0,-1.5,-6 --target 0,0,10 --up 0,-1,0 --fov 60 --size 1280x720 -o view.png
//
// --colmap reads cameras.txt and images.txt (COLMAP's text model, PINHOLE cameras as written by
// image_undistorter) and renders exactly that view. --stats prints visible splats and timings.
#include <charconv>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>

#include "otrv/splat.hpp"

namespace splat = otrv::splat;

namespace {

int usage() {
  std::cerr << "usage: otrv-render <scene.ply> (--colmap <sparse dir> --image <name> | --eye x,y,z --target x,y,z\n"
               "                   [--up x,y,z] [--fov deg] [--size WxH]) [-o out.png] [--bg r,g,b] [--threads n]\n"
               "                   [--stats]\n";
  return 2;
}

std::optional<splat::Vec3> parse_vec(const std::string& s) {
  splat::Vec3 v{};
  std::istringstream ss(s);
  char comma = 0;
  if (!(ss >> v[0] >> comma >> v[1] >> comma >> v[2])) return std::nullopt;
  return v;
}

// One view from a COLMAP text model.
splat::Camera colmap_view(const std::string& dir, const std::string& name) {
  struct Intr {
    int w, h;
    double fx, fy, cx, cy;
  };
  std::map<int, Intr> cams;
  std::ifstream cf(dir + "/cameras.txt");
  if (!cf) throw std::runtime_error("cannot read " + dir + "/cameras.txt");
  for (std::string line; std::getline(cf, line);) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    int id = 0, w = 0, h = 0;
    std::string model;
    ss >> id >> model >> w >> h;
    if (model == "PINHOLE") {
      Intr in{w, h, 0, 0, 0, 0};
      ss >> in.fx >> in.fy >> in.cx >> in.cy;
      cams[id] = in;
    } else if (model == "SIMPLE_PINHOLE") {
      Intr in{w, h, 0, 0, 0, 0};
      ss >> in.fx >> in.cx >> in.cy;
      in.fy = in.fx;
      cams[id] = in;
    }
  }
  std::ifstream imf(dir + "/images.txt");
  if (!imf) throw std::runtime_error("cannot read " + dir + "/images.txt");
  bool pose_line = true;
  for (std::string line; std::getline(imf, line);) {
    if (!line.empty() && line[0] == '#') continue;
    if (!pose_line) {  // POINTS2D line
      pose_line = true;
      continue;
    }
    pose_line = false;
    std::istringstream ss(line);
    int id = 0, cam = 0;
    std::array<double, 4> q{};
    splat::Vec3 t{};
    std::string img;
    if (!(ss >> id >> q[0] >> q[1] >> q[2] >> q[3] >> t[0] >> t[1] >> t[2] >> cam >> img)) continue;
    if (img != name) continue;
    const auto it = cams.find(cam);
    if (it == cams.end())
      throw std::runtime_error("camera " + std::to_string(cam) + " is not PINHOLE (undistort first)");
    const Intr& c = it->second;
    return splat::Camera::from_colmap(q, t, c.w, c.h, c.fx, c.fy, c.cx, c.cy);
  }
  throw std::runtime_error("no image named " + name + " in " + dir + "/images.txt");
}

}  // namespace

int main(int argc, char** argv) {
  std::string ply, out = "render.png", colmap, image;
  std::optional<splat::Vec3> eye, target;
  splat::Vec3 up{0, -1, 0};  // COLMAP scenes: y points down
  double fov = 60;
  int w = 1280, h = 720;
  bool stats = false;
  splat::RenderOptions opts;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      const auto next = [&]() -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
        return argv[++i];
      };
      if (a == "-o" || a == "--out")
        out = next();
      else if (a == "--colmap")
        colmap = next();
      else if (a == "--image")
        image = next();
      else if (a == "--eye")
        eye = parse_vec(next());
      else if (a == "--target")
        target = parse_vec(next());
      else if (a == "--up") {
        const auto v = parse_vec(next());
        if (!v) return usage();
        up = *v;
      } else if (a == "--fov")
        fov = std::stod(next());
      else if (a == "--size") {
        if (std::sscanf(next().c_str(), "%dx%d", &w, &h) != 2) return usage();
      } else if (a == "--bg") {
        const auto v = parse_vec(next());
        if (!v) return usage();
        opts.background = {static_cast<float>((*v)[0]), static_cast<float>((*v)[1]), static_cast<float>((*v)[2])};
      } else if (a == "--threads")
        opts.threads = std::stoi(next());
      else if (a == "--stats")
        stats = true;
      else if (a == "-h" || a == "--help")
        return usage();
      else if (ply.empty() && a[0] != '-')
        ply = a;
      else
        return usage();
    }
    if (ply.empty()) return usage();

    splat::Camera cam;
    if (!colmap.empty()) {
      if (image.empty()) return usage();
      cam = colmap_view(colmap, image);
    } else if (eye && target) {
      cam = splat::Camera::look_at(*eye, *target, up, fov, w, h);
    } else {
      return usage();
    }

    const auto g = splat::load_ply(ply);
    splat::RenderStats st;
    const auto img = splat::render(g, cam, opts, &st);
    splat::write_png(out, img.width, img.height, img.to_rgb8());
    if (stats) {
      std::printf("splats      %zu (%zu visible, SH degree %d)\n", g.size(), st.visible, g.sh_degree);
      std::printf("image       %dx%d -> %s\n", img.width, img.height, out.c_str());
      std::printf("tile hits   %zu\n", st.tile_hits);
      std::printf("time        project %.1f ms, sort %.1f ms, raster %.1f ms\n", st.project_ms, st.sort_ms,
                  st.raster_ms);
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "otrv-render: " << e.what() << "\n";
    return 1;
  }
}
