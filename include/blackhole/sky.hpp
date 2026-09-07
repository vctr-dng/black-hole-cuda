#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#ifndef BH_HD
#ifdef __CUDACC__
#define BH_HD __host__ __device__
#else
#define BH_HD
#endif
#endif

namespace blackhole::sky {

/// Linear-RGB color or world-space direction (x right, y up, z forward).
struct Vec3 {
  float x, y, z;
};

constexpr float kPi = 3.14159265358979323846f;

/// Direction -> equirectangular (u, v). u in [0, 1) with +z at u = 0.5, +x at
/// 0.75; v = 0 is zenith (+y), v = 1 is nadir. `dir` must be unit length
/// (dir.y is clamped to [-1, 1] before asin).
BH_HD inline void dirToEquirect(Vec3 dir, float &u, float &v) {
  u = ::atan2f(dir.x, dir.z) / (2.0f * kPi) + 0.5f;
  if (u >= 1.0f)
    u -= 1.0f;
  if (u < 0.0f)
    u = 0.0f;
  const float y = dir.y < -1.0f ? -1.0f : (dir.y > 1.0f ? 1.0f : dir.y);
  v = 0.5f - ::asinf(y) / kPi;
}

/// Inverse of dirToEquirect; returns a unit vector.
BH_HD inline Vec3 equirectToDir(float u, float v) {
  const float lon = (u - 0.5f) * 2.0f * kPi;
  const float lat = (0.5f - v) * kPi;
  const float c = ::cosf(lat);
  return {c * ::sinf(lon), ::sinf(lat), c * ::cosf(lon)};
}

namespace detail {
BH_HD inline std::uint32_t hash32(std::uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}
BH_HD inline std::uint32_t hashCell(int ix, int iy, int iz, std::uint32_t k) {
  std::uint32_t h = hash32(static_cast<std::uint32_t>(ix) * 0x9e3779b1U + k);
  h = hash32(h ^ (static_cast<std::uint32_t>(iy) * 0x85ebca6bU));
  h = hash32(h ^ (static_cast<std::uint32_t>(iz) * 0xc2b2ae35U));
  return h;
}
BH_HD inline float toUnit(std::uint32_t h) {
  return static_cast<float>(h >> 8) * (1.0f / 16777216.0f);
}
} // namespace detail

/// Procedural starfield + faint galactic band. Pure function of direction.
BH_HD inline Vec3 starfield(Vec3 dir) {
  const float len = ::sqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
  const float inv = len > 0.0f ? 1.0f / len : 0.0f;
  const Vec3 d{dir.x * inv, dir.y * inv, dir.z * inv};

  constexpr float kScale = 100.0f; // shell radius in cells
  constexpr float kBand = 0.5f;    // stars live in |r - kScale| < kBand
  constexpr float kProb = 0.04f;
  const float px = d.x * kScale, py = d.y * kScale, pz = d.z * kScale;
  const int cx = static_cast<int>(::floorf(px));
  const int cy = static_cast<int>(::floorf(py));
  const int cz = static_cast<int>(::floorf(pz));

  float r = 0.0f, g = 0.0f, b = 0.0f;
  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        const int ix = cx + dx, iy = cy + dy, iz = cz + dz;
        const std::uint32_t h0 = detail::hashCell(ix, iy, iz, 1u);
        if (detail::toUnit(h0) >= kProb)
          continue;
        const float sx = static_cast<float>(ix) +
                         detail::toUnit(detail::hashCell(ix, iy, iz, 2u));
        const float sy = static_cast<float>(iy) +
                         detail::toUnit(detail::hashCell(ix, iy, iz, 3u));
        const float sz = static_cast<float>(iz) +
                         detail::toUnit(detail::hashCell(ix, iy, iz, 4u));
        const float sl = ::sqrtf(sx * sx + sy * sy + sz * sz);
        if (::fabsf(sl - kScale) >= kBand)
          continue;
        const float inl = 1.0f / sl;
        const float crx = d.y * sz * inl - d.z * sy * inl;
        const float cry = d.z * sx * inl - d.x * sz * inl;
        const float crz = d.x * sy * inl - d.y * sx * inl;
        const float cosang = (d.x * sx + d.y * sy + d.z * sz) * inl;
        if (cosang <= 0.0f)
          continue;
        const float ang2 = crx * crx + cry * cry +
                           crz * crz; // ~ angle^2 (cross: no cancellation)
        const float rad =
            (0.10f + 0.15f * detail::toUnit(detail::hashCell(ix, iy, iz, 5u))) *
            (kPi / 180.0f);
        const float t2 = ang2 / (rad * rad);
        if (t2 > 9.0f)
          continue;
        const float u1 = detail::toUnit(detail::hashCell(ix, iy, iz, 6u));
        const float bright = 0.4f + 3.0f * u1 * u1 * u1;
        const float temp = detail::toUnit(detail::hashCell(ix, iy, iz, 7u));
        const float w = bright * ::expf(-0.5f * t2);
        // blue (cool=0) -> white -> orange (1)
        const float tr = 0.7f + 0.6f * temp, tb = 1.3f - 0.7f * temp;
        const float tg = 1.0f - 0.2f * (temp - 0.5f) * (temp - 0.5f);
        r += w * tr;
        g += w * tg;
        b += w * tb;
      }
    }
  }
  // Galactic band: great circle with normal n, faint warm glow.
  const float nx = 0.3f, ny = 0.8f, nz = 0.52f;
  const float nl = ::sqrtf(nx * nx + ny * ny + nz * nz);
  const float s = (d.x * nx + d.y * ny + d.z * nz) / nl;
  const float glow = 0.012f * ::expf(-s * s / (2.0f * 0.12f * 0.12f));
  r += glow * 1.0f;
  g += glow * 0.9f;
  b += glow * 0.8f;
  return {r, g, b};
}

/// Linear-RGB float image, row-major, 3 floats per pixel, row 0 = top.
struct SkyImage {
  int width = 0;
  int height = 0;
  std::vector<float> rgb;
};

/// Loads PNG/JPG (8-bit sRGB -> linear) or Radiance .hdr (float, as is).
/// Throws std::runtime_error (message contains `path`) on failure.
SkyImage loadSkyImage(const std::string &path);

/// r = x/(w-1), g = y/(h-1), b = 0.25 (w, h >= 2).
SkyImage makeSyntheticSky(int w, int h);

/// CPU reference for the GPU texture: bilinear in (u, v), texel centers at
/// (i + 0.5) / size, wrap in u, clamp in v.
Vec3 sampleSky(const SkyImage &image, Vec3 dir);

/// RAII float4 cudaArray + texture object (linear filter, normalized coords,
/// wrap u, clamp v). Throws std::invalid_argument for empty/ill-sized images,
/// std::runtime_error on CUDA errors.
class SkyTexture {
public:
  explicit SkyTexture(const SkyImage &image);
  ~SkyTexture();
  SkyTexture(const SkyTexture &) = delete;
  SkyTexture &operator=(const SkyTexture &) = delete;
  SkyTexture(SkyTexture &&other) noexcept;
  SkyTexture &operator=(SkyTexture &&other) noexcept;

  /// 0 for a moved-from object.
  cudaTextureObject_t texture() const { return texture_; }

private:
  void release() noexcept;
  cudaArray_t array_ = nullptr;
  cudaTextureObject_t texture_ = 0;
};

#ifdef __CUDACC__
/// Device-only: sample the sky texture in direction `dir`.
__device__ inline Vec3 sampleSkyDevice(cudaTextureObject_t tex, Vec3 dir) {
  float u, v;
  dirToEquirect(dir, u, v);
  const float4 c = tex2D<float4>(tex, u, v);
  return {c.x, c.y, c.z};
}
#endif

} // namespace blackhole::sky
