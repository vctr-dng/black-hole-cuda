#pragma once

// Post-processing math: tone mapping, sRGB encoding, supersampling offsets,
// bloom weights.  Header-only; BH_HD functions are usable on host and device.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#ifndef BH_HD
#if defined(__CUDACC__)
#define BH_HD __host__ __device__
#else
#define BH_HD
#endif
#endif

namespace blackhole::post {

struct Vec3 {
  float x, y, z;
};

BH_HD inline Vec3 operator+(Vec3 a, Vec3 b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
BH_HD inline Vec3 operator-(Vec3 a, Vec3 b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
BH_HD inline Vec3 operator*(Vec3 a, float s) {
  return {a.x * s, a.y * s, a.z * s};
}

BH_HD inline float clamp01(float x) { return fminf(fmaxf(x, 0.0f), 1.0f); }

// Narkowicz ACES fit.  Input is clamped to [0, 1e4] (NaN -> 0) so that
// inf / FLT_MAX cannot produce inf/inf = NaN.
BH_HD inline float acesChannel(float x) {
  x = fminf(fmaxf(x, 0.0f), 1.0e4f);
  const float num = x * (2.51f * x + 0.03f);
  const float den = x * (2.43f * x + 0.59f) + 0.14f;
  return clamp01(num / den);
}

BH_HD inline Vec3 acesFilm(Vec3 c) {
  return {acesChannel(c.x), acesChannel(c.y), acesChannel(c.z)};
}

BH_HD inline float linearToSrgb(float x) {
  x = clamp01(x);
  return x <= 0.0031308f ? 12.92f * x : 1.055f * powf(x, 1.0f / 2.4f) - 0.055f;
}

BH_HD inline float srgbToLinear(float x) {
  x = clamp01(x);
  return x <= 0.04045f ? x / 12.92f : powf((x + 0.055f) / 1.055f, 2.4f);
}

BH_HD inline std::uint8_t toByte(float v) {
  return static_cast<std::uint8_t>(roundf(255.0f * clamp01(v)));
}

BH_HD inline Vec3 postProcessPixel(Vec3 c, float exposure) {
  const Vec3 t = acesFilm(c * exposure);
  return {linearToSrgb(t.x), linearToSrgb(t.y), linearToSrgb(t.z)};
}

BH_HD inline void packRgba8(Vec3 s, std::uint8_t* out4) {
  out4[0] = toByte(s.x);
  out4[1] = toByte(s.y);
  out4[2] = toByte(s.z);
  out4[3] = 255;
}

struct SampleOffset {
  float dx, dy;
};

// Integer square root of a positive value, exact for all 64-bit inputs used
// here (corrected after the floating-point estimate).
BH_HD inline long long isqrtExact(long long v) {
  long long r = static_cast<long long>(sqrt(static_cast<double>(v)));
  while (r * r > v) --r;
  while ((r + 1) * (r + 1) <= v) ++r;
  return r;
}

BH_HD inline bool isValidSpp(int spp) {
  if (spp <= 0) return false;
  const long long r = isqrtExact(static_cast<long long>(spp));
  return r * r == static_cast<long long>(spp);
}

// splitmix64 finaliser.
BH_HD inline std::uint64_t mix64(std::uint64_t z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// Stratified jittered offset inside the pixel, in [0, 1).  Deterministic in
// (sampleIndex, spp, px, py).  Requires isValidSpp(spp).
BH_HD inline SampleOffset sampleOffset(int sampleIndex, int spp, int px,
                                       int py) {
  if (spp == 1) return {0.5f, 0.5f};
  const long long n = isqrtExact(static_cast<long long>(spp));
  const long long idx = static_cast<long long>(sampleIndex);
  const long long cx = idx % n;
  const long long cy = idx / n;
  const std::uint64_t key =
      (static_cast<std::uint64_t>(static_cast<std::uint32_t>(px)) << 32) |
      static_cast<std::uint64_t>(static_cast<std::uint32_t>(py));
  const std::uint64_t h1 = mix64(key ^ mix64(static_cast<std::uint64_t>(idx)));
  const std::uint64_t h2 = mix64(h1 ^ 0xD1B54A32D192ED03ull);
  const float u = static_cast<float>(static_cast<std::uint32_t>(h1 >> 40)) *
                  (1.0f / 16777216.0f);
  const float v = static_cast<float>(static_cast<std::uint32_t>(h2 >> 40)) *
                  (1.0f / 16777216.0f);
  const float fn = static_cast<float>(n);
  const float kMaxBelowOne = 0.99999994f;  // largest float < 1
  const float dx = fminf((static_cast<float>(cx) + u) / fn, kMaxBelowOne);
  const float dy = fminf((static_cast<float>(cy) + v) / fn, kMaxBelowOne);
  return {dx, dy};
}

inline int gaussianKernelRadius(float sigma) {
  const int r = static_cast<int>(std::ceil(3.0f * sigma));
  return r < 1 ? 1 : r;
}

inline std::vector<float> gaussianKernel1D(float sigma) {
  if (!(sigma > 0.0f) || !std::isfinite(sigma))
    throw std::invalid_argument("gaussianKernel1D: sigma must be finite > 0");
  const int r = gaussianKernelRadius(sigma);
  const double s = static_cast<double>(sigma);
  std::vector<double> w(static_cast<std::size_t>(2 * r + 1));
  double sum = 0.0;
  for (int i = -r; i <= r; ++i) {
    const double x = static_cast<double>(i);
    const double e = std::exp(-(x * x) / (2.0 * s * s));
    w[static_cast<std::size_t>(i + r)] = e;
    sum += e;
  }
  std::vector<float> k(w.size());
  for (std::size_t i = 0; i < w.size(); ++i) k[i] = static_cast<float>(w[i] / sum);
  return k;
}

// Clamp-to-edge 1-D convolution.
inline std::vector<float> blurRow(const std::vector<float>& row,
                                  const std::vector<float>& kernel) {
  const long long n = static_cast<long long>(row.size());
  const long long m = static_cast<long long>(kernel.size());
  const long long r = (m - 1) / 2;
  std::vector<float> out(row.size(), 0.0f);
  for (long long i = 0; i < n; ++i) {
    double acc = 0.0;
    for (long long j = 0; j < m; ++j) {
      long long s = i + j - r;
      s = s < 0 ? 0 : (s >= n ? n - 1 : s);
      acc += static_cast<double>(kernel[static_cast<std::size_t>(j)]) *
             static_cast<double>(row[static_cast<std::size_t>(s)]);
    }
    out[static_cast<std::size_t>(i)] = static_cast<float>(acc);
  }
  return out;
}

BH_HD inline Vec3 bloomCombine(Vec3 scene, Vec3 blurred, float strength) {
  return scene + blurred * strength;
}

}  // namespace blackhole::post
