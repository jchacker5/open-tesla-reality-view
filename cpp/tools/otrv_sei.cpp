// otrv-sei: print a TeslaCam clip's SEI telemetry as JSON (same output as `python -m tsplat.sei`).
//
//   otrv-sei 2026-11-20_07-20-00-front.mp4 > telemetry.json
//   otrv-sei --summary clip.mp4      # frame count, speed range, GPS fix rate, decode time
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iostream>

#include "otrv/sei.hpp"

namespace {

int usage() {
  std::cerr << "usage: otrv-sei [--summary] <clip.mp4>\n";
  return 2;
}

void summarize(const std::vector<otrv::sei::Frame>& rows, double ms) {
  double vmin = 1e300, vmax = -1e300;
  std::size_t gps = 0, speed = 0;
  for (const auto& f : rows) {
    if (const auto v = f.speed_mps()) {
      vmin = std::min(vmin, *v);
      vmax = std::max(vmax, *v);
      ++speed;
    }
    const auto lat = f.latitude_deg(), lon = f.longitude_deg();
    if (lat && lon && (*lat != 0.0 || *lon != 0.0)) ++gps;
  }
  std::printf("frames      %zu\n", rows.size());
  if (speed != 0) std::printf("speed       %.1f - %.1f mph\n", vmin * 2.23694, vmax * 2.23694);
  std::printf("gps fixes   %zu / %zu\n", gps, rows.size());
  std::printf("decode      %.1f ms\n", ms);
}

}  // namespace

int main(int argc, char** argv) {
  bool summary = false;
  const char* path = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--summary") == 0)
      summary = true;
    else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0)
      return usage();
    else if (path == nullptr)
      path = argv[i];
    else
      return usage();
  }
  if (path == nullptr) return usage();

  try {
    const auto t0 = std::chrono::steady_clock::now();
    const auto rows = otrv::sei::extract_file(path);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (summary) {
      summarize(rows, ms);
      return 0;
    }
    if (rows.empty()) {
      std::cerr << "No SEI metadata (needs firmware 2025.44.25+, HW3+, car not parked)\n";
      return 1;
    }
    std::cout << otrv::sei::to_json(rows);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "otrv-sei: " << e.what() << "\n";
    return 1;
  }
}
