#pragma once

// Accretion-disk math (track 4a), header-only and usable from host and device.
//
// Units: rs = 1, c = 1, M = 0.5. World axes: x right, y up, z forward. The
// accretion disk lies in the plane y = 0 and orbits counter-clockwise seen
// from +y.

#include <cmath>

#ifndef BH_HD
#ifdef __CUDACC__
#define BH_HD __host__ __device__
#else
#define BH_HD
#endif
#endif
namespace blackhole::disk {
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
BH_HD inline Vec3 operator*(float s, Vec3 a) { return a * s; }
BH_HD inline float dot(Vec3 a, Vec3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
BH_HD inline Vec3 cross(Vec3 a, Vec3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
BH_HD inline float length(Vec3 a) { return sqrtf(dot(a, a)); }
BH_HD inline Vec3 normalize(Vec3 a) { return a * (1.0f / length(a)); }
struct DiskParams {
  float rInner = 3.0f;
  float rOuter = 12.0f;
  float tInnerK = 12000.0f;
  float brightness = 1.0f;
};
BH_HD inline bool inDisk(float r, const DiskParams &p) {
  return r >= p.rInner && r <= p.rOuter;
}
BH_HD inline float diskTemperature(float r, const DiskParams &p) {
  return p.tInnerK * powf(r / p.rInner, -0.75f);
}
BH_HD inline float orbitalSpeedLocal(float r) {
  return sqrtf(0.5f / (r - 1.0f));
}
BH_HD inline Vec3 orbitalDirection(Vec3 pos) {
  return normalize(Vec3{pos.z, 0.0f, -pos.x});
}
BH_HD inline float gFactor(float r, float c) {
  const float v = orbitalSpeedLocal(r);
  const float gamma = 1.0f / sqrtf(1.0f - v * v);
  return sqrtf(1.0f - 1.0f / r) / (gamma * (1.0f - v * c));
}
// Exact static-frame cosine between the photon direction (reversed) and the
// orbital direction at `pos`: -h.y * sqrt(1 - 1/r) / r, with h = pos x vel.
BH_HD inline float dopplerCosine(const Vec3 &pos, const Vec3 &vel) {
  const Vec3 h = cross(pos, vel);
  const float r = length(pos);
  return -h.y * sqrtf(1.0f - 1.0f / r) / r;
}
BH_HD inline float gFactorForRay(Vec3 pos, Vec3 rayDir) {
  return gFactor(length(pos), dot(orbitalDirection(pos), rayDir * -1.0f));
}
BH_HD inline float intensityBoost(float g) { return g * g * g; }
#define BH_LUT_BODY                                                            \
  {1.000000f, 0.028693f, 0.000000f}, {1.000000f, 0.041564f, 0.000000f},        \
      {1.000000f, 0.055772f, 0.000000f}, {1.000000f, 0.071315f, 0.000000f},    \
      {1.000000f, 0.088196f, 0.000000f}, {1.000000f, 0.106413f, 0.000000f},    \
      {1.000000f, 0.125966f, 0.000000f}, {1.000000f, 0.146854f, 0.000000f},    \
      {1.000000f, 0.169067f, 0.000000f}, {1.000000f, 0.192594f, 0.000000f},    \
      {1.000000f, 0.217418f, 0.000000f}, {1.000000f, 0.243513f, 0.000000f},    \
      {1.000000f, 0.270847f, 0.009411f}, {1.000000f, 0.299379f, 0.021526f},    \
      {1.000000f, 0.329058f, 0.036390f}, {1.000000f, 0.359825f, 0.054321f},    \
      {1.000000f, 0.391610f, 0.075635f}, {1.000000f, 0.424334f, 0.100632f},    \
      {1.000000f, 0.457905f, 0.129593f}, {1.000000f, 0.492223f, 0.162769f},    \
      {1.000000f, 0.527175f, 0.200376f}, {1.000000f, 0.562640f, 0.242582f},    \
      {1.000000f, 0.598487f, 0.289503f}, {1.000000f, 0.634575f, 0.341196f},    \
      {1.000000f, 0.670759f, 0.397649f}, {1.000000f, 0.706885f, 0.458780f},    \
      {1.000000f, 0.742800f, 0.524433f}, {1.000000f, 0.778349f, 0.594379f},    \
      {1.000000f, 0.813379f, 0.668313f}, {1.000000f, 0.847744f, 0.745865f},    \
      {1.000000f, 0.881308f, 0.826605f}, {1.000000f, 0.913945f, 0.910053f},    \
      {1.000000f, 0.945544f, 0.995688f}, {0.923388f, 0.901237f, 1.000000f},    \
      {0.853724f, 0.858223f, 1.000000f}, {0.793494f, 0.819886f, 1.000000f},    \
      {0.741198f, 0.785629f, 1.000000f}, {0.695612f, 0.754947f, 1.000000f},    \
      {0.655725f, 0.727405f, 1.000000f}, {0.620700f, 0.702634f, 1.000000f},    \
      {0.589842f, 0.680313f, 1.000000f}, {0.562567f, 0.660164f, 1.000000f},    \
      {0.538385f, 0.641946f, 1.000000f}, {0.516884f, 0.625446f, 1.000000f},    \
      {0.497711f, 0.610480f, 1.000000f}, {0.480571f, 0.596886f, 1.000000f},    \
      {0.465206f, 0.584519f, 1.000000f}, {0.451401f, 0.573254f, 1.000000f},    \
      {0.438967f, 0.562977f, 1.000000f}, {0.427743f, 0.553591f, 1.000000f},    \
      {0.417589f, 0.545007f, 1.000000f}, {0.408384f, 0.537145f, 1.000000f},    \
      {0.400023f, 0.529938f, 1.000000f}, {0.392414f, 0.523322f, 1.000000f},    \
      {0.385477f, 0.517242f, 1.000000f}, {0.379142f, 0.511648f, 1.000000f},    \
      {0.373346f, 0.506496f, 1.000000f}, {0.368036f, 0.501745f, 1.000000f},    \
      {0.363164f, 0.497360f, 1.000000f}, {0.358686f, 0.493309f, 1.000000f},    \
      {0.354566f, 0.489563f, 1.000000f}, {0.350770f, 0.486095f, 1.000000f},    \
      {0.347269f, 0.482882f, 1.000000f}, {0.344034f, 0.479903f, 1.000000f},
// Linear-sRGB blackbody chromaticity (max channel 1), 64 log-spaced samples
// over [1000 K, 40000 K], from CIE 1931 integration.
static const float kLutHost[64][3] = {BH_LUT_BODY};
#ifdef __CUDACC__
static __constant__ float kLutDev[64][3] = {BH_LUT_BODY};
#endif
#undef BH_LUT_BODY

BH_HD inline Vec3 blackbodyColor(float t) {
#ifdef __CUDA_ARCH__
  const auto &kLut = kLutDev;
#else
  const auto &kLut = kLutHost;
#endif
  if (!(t > 1000.0f))
    t = 1000.0f;
  if (t > 40000.0f)
    t = 40000.0f;
  float f = logf(t / 1000.0f) / logf(40.0f) * 63.0f;
  int i = static_cast<int>(f);
  if (i > 62)
    i = 62;
  const float w = f - static_cast<float>(i);
  Vec3 c{kLut[i][0] + w * (kLut[i + 1][0] - kLut[i][0]),
         kLut[i][1] + w * (kLut[i + 1][1] - kLut[i][1]),
         kLut[i][2] + w * (kLut[i + 1][2] - kLut[i][2])};
  const float m = fmaxf(c.x, fmaxf(c.y, c.z));
  return c * (1.0f / m);
}
BH_HD inline float radiantBrightness(float tK, const DiskParams &p) {
  const float q = tK / p.tInnerK;
  return p.brightness * q * q * q * q;
}
BH_HD inline Vec3 diskRadiance(float r, float c, const DiskParams &p) {
  if (!inDisk(r, p))
    return Vec3{0, 0, 0};
  const float g = gFactor(r, c);
  const float t = diskTemperature(r, p);
  return blackbodyColor(g * t) * (radiantBrightness(t, p) * intensityBoost(g));
}
struct Crossing {
  bool hit;
  Vec3 pos;
  float t;
};
BH_HD inline Crossing planeCrossing(Vec3 p0, Vec3 p1) {
  const bool hit = (p0.y < 0 && p1.y >= 0) || (p0.y > 0 && p1.y <= 0);
  if (!hit)
    return {false, p0, 0.0f};
  const float t = p0.y / (p0.y - p1.y);
  Vec3 pos = p0 + (p1 - p0) * t;
  pos.y = 0.0f;
  return {true, pos, t};
}
} // namespace blackhole::disk
