// CPU-side tests for the sky module: equirectangular mapping, procedural
// starfield, image loading and the CPU bilinear reference sampler.
#include "blackhole/sky.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "stb_image_write.h"  // implementation lives in src/image.cpp

namespace sky = blackhole::sky;
using sky::SkyImage;
using sky::Vec3;

namespace {

constexpr double kPi = 3.14159265358979323846;

// Deterministic across standard libraries: raw mt19937 output only.
class Rng {
 public:
  explicit Rng(std::uint32_t seed) : gen_(seed) {}
  double uniform() { return static_cast<double>(gen_() >> 8) / 16777216.0; }

 private:
  std::mt19937 gen_;
};

Vec3 normalized(double x, double y, double z) {
  const double n = std::sqrt(x * x + y * y + z * z);
  return {static_cast<float>(x / n), static_cast<float>(y / n),
          static_cast<float>(z / n)};
}

// Uniform on the sphere (Archimedes): y uniform in [-1, 1], azimuth uniform.
Vec3 randomDir(Rng& rng) {
  const double y = 2.0 * rng.uniform() - 1.0;
  const double phi = 2.0 * kPi * rng.uniform();
  const double s = std::sqrt(std::max(0.0, 1.0 - y * y));
  return {static_cast<float>(s * std::sin(phi)), static_cast<float>(y),
          static_cast<float>(s * std::cos(phi))};
}

// Test-local copy of the equirect convention, independent of the code under
// test: u = lon/(2 pi) + 0.5 with lon = atan2(x, z); v = 0.5 - lat/pi.
Vec3 dirFromUV(double u, double v) {
  const double lon = (u - 0.5) * 2.0 * kPi;
  const double lat = (0.5 - v) * kPi;
  return {static_cast<float>(std::cos(lat) * std::sin(lon)),
          static_cast<float>(std::sin(lat)),
          static_cast<float>(std::cos(lat) * std::cos(lon))};
}

double luminance(Vec3 c) { return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z; }

// `d` rotated by `angle` radians towards a random tangent direction.
Vec3 perturbed(Vec3 d, double angle, Rng& rng) {
  const Vec3 g = randomDir(rng);
  const double dot = g.x * d.x + g.y * d.y + g.z * d.z;
  double tx = g.x - dot * d.x, ty = g.y - dot * d.y, tz = g.z - dot * d.z;
  const double tn = std::sqrt(tx * tx + ty * ty + tz * tz);
  tx /= tn;
  ty /= tn;
  tz /= tn;
  const double c = std::cos(angle), s = std::sin(angle);
  return normalized(c * d.x + s * tx, c * d.y + s * ty, c * d.z + s * tz);
}

// ---------------------------------------------------------------------------
// Equirectangular mapping
// ---------------------------------------------------------------------------

TEST(EquirectTest, CardinalDirections) {
  float u = -1, v = -1;
  sky::dirToEquirect({0, 0, 1}, u, v);
  EXPECT_NEAR(u, 0.5f, 1e-6f);
  EXPECT_NEAR(v, 0.5f, 1e-6f);
  sky::dirToEquirect({1, 0, 0}, u, v);
  EXPECT_NEAR(u, 0.75f, 1e-6f);
  EXPECT_NEAR(v, 0.5f, 1e-6f);
  sky::dirToEquirect({-1, 0, 0}, u, v);
  EXPECT_NEAR(u, 0.25f, 1e-6f);
  EXPECT_NEAR(v, 0.5f, 1e-6f);
  sky::dirToEquirect({0, 1, 0}, u, v);  // zenith: v = 0 (top row)
  EXPECT_NEAR(v, 0.0f, 1e-6f);
  sky::dirToEquirect({0, -1, 0}, u, v);  // nadir: v = 1
  EXPECT_NEAR(v, 1.0f, 1e-6f);
}

TEST(EquirectTest, ElevationIsLinearInV) {
  float u, v;
  sky::dirToEquirect(normalized(0, 0.5, std::sqrt(0.75)), u, v);  // +30 deg
  EXPECT_NEAR(v, 1.0 / 3.0, 1e-6);
  sky::dirToEquirect(normalized(0, -0.5, std::sqrt(0.75)), u, v);  // -30 deg
  EXPECT_NEAR(v, 2.0 / 3.0, 1e-6);
}

TEST(EquirectTest, MinusZIsTheWrapSeam) {
  float u, v;
  sky::dirToEquirect({0, 0, -1}, u, v);
  // -z is exactly the seam: u may be reported as 0 or (just below) 1.
  EXPECT_TRUE(u < 1e-6f || u > 1.0f - 1e-6f) << "u=" << u;
  EXPECT_LT(u, 1.0f);  // range is [0, 1)

  // Slightly to the +x side of -z -> just below 1; -x side -> just above 0.
  sky::dirToEquirect(normalized(1e-3, 0, -1), u, v);
  EXPECT_NEAR(u, 1.0 - 1e-3 / (2 * kPi), 1e-6);
  sky::dirToEquirect(normalized(-1e-3, 0, -1), u, v);
  EXPECT_NEAR(u, 1e-3 / (2 * kPi), 1e-6);
}

TEST(EquirectTest, UStaysInHalfOpenUnitInterval) {
  Rng rng(1);
  float uMin = 1, uMax = 0, vMin = 1, vMax = 0;
  for (int i = 0; i < 20000; ++i) {
    float u, v;
    sky::dirToEquirect(randomDir(rng), u, v);
    ASSERT_GE(u, 0.0f);
    ASSERT_LT(u, 1.0f);
    ASSERT_GE(v, 0.0f);
    ASSERT_LE(v, 1.0f);
    uMin = std::min(uMin, u);
    uMax = std::max(uMax, u);
    vMin = std::min(vMin, v);
    vMax = std::max(vMax, v);
  }
  // The whole square is covered (guards against a constant mapping).
  EXPECT_LT(uMin, 0.01f);
  EXPECT_GT(uMax, 0.99f);
  EXPECT_LT(vMin, 0.01f);
  EXPECT_GT(vMax, 0.99f);
}

TEST(EquirectTest, PolesAreClampedAndFinite) {
  float u, v;
  // y slightly out of [-1, 1] (unnormalized input / rounding) must not NaN.
  sky::dirToEquirect({0, 1.0001f, 0}, u, v);
  EXPECT_TRUE(std::isfinite(u));
  EXPECT_NEAR(v, 0.0f, 1e-6f);
  sky::dirToEquirect({0, -2.0f, 0}, u, v);
  EXPECT_TRUE(std::isfinite(u));
  EXPECT_NEAR(v, 1.0f, 1e-6f);
  // Exactly on the axis: azimuth is undefined but must be a valid u.
  sky::dirToEquirect({0, 1, 0}, u, v);
  EXPECT_TRUE(std::isfinite(u));
  EXPECT_GE(u, 0.0f);
  EXPECT_LT(u, 1.0f);
}

TEST(EquirectTest, EquirectToDirCardinalsAndUnitLength) {
  auto expectDir = [](float u, float v, Vec3 want) {
    const Vec3 d = sky::equirectToDir(u, v);
    EXPECT_NEAR(d.x, want.x, 1e-6f) << u << "," << v;
    EXPECT_NEAR(d.y, want.y, 1e-6f) << u << "," << v;
    EXPECT_NEAR(d.z, want.z, 1e-6f) << u << "," << v;
  };
  expectDir(0.5f, 0.5f, {0, 0, 1});
  expectDir(0.75f, 0.5f, {1, 0, 0});
  expectDir(0.25f, 0.5f, {-1, 0, 0});
  expectDir(0.5f, 0.0f, {0, 1, 0});   // v = 0 is up
  expectDir(0.5f, 1.0f, {0, -1, 0});  // v = 1 is down
  expectDir(0.0f, 0.5f, {0, 0, -1});

  Rng rng(2);
  for (int i = 0; i < 1000; ++i) {
    const Vec3 d = sky::equirectToDir(static_cast<float>(rng.uniform()),
                                      static_cast<float>(rng.uniform()));
    EXPECT_NEAR(d.x * d.x + d.y * d.y + d.z * d.z, 1.0f, 1e-5f);
  }
}

TEST(EquirectTest, RoundTripRandomDirections) {
  // Float atan2/asin/sin/cos each carry ~1e-7 error; 1e-5 leaves a 100x margin
  // yet still catches a wrong constant (e.g. pi vs 2*pi) immediately. Poles
  // are excluded because azimuth conditioning degrades as 1/cos(lat).
  Rng rng(3);
  int checked = 0;
  while (checked < 20000) {
    const Vec3 d = randomDir(rng);
    if (std::fabs(d.y) > 0.99f) continue;
    ++checked;
    float u, v;
    sky::dirToEquirect(d, u, v);
    const Vec3 r = sky::equirectToDir(u, v);
    ASSERT_NEAR(r.x, d.x, 1e-5f);
    ASSERT_NEAR(r.y, d.y, 1e-5f);
    ASSERT_NEAR(r.z, d.z, 1e-5f);
  }
}

// ---------------------------------------------------------------------------
// Procedural starfield
// ---------------------------------------------------------------------------

constexpr int kStarSamples = 200000;

struct StarSamples {
  std::vector<Vec3> dirs;
  std::vector<Vec3> colors;
};

const StarSamples& starSamples() {
  static const StarSamples samples = [] {
    StarSamples s;
    Rng rng(12345);
    s.dirs.reserve(kStarSamples);
    s.colors.reserve(kStarSamples);
    for (int i = 0; i < kStarSamples; ++i) {
      const Vec3 d = randomDir(rng);
      s.dirs.push_back(d);
      s.colors.push_back(sky::starfield(d));
    }
    return s;
  }();
  return samples;
}

TEST(StarfieldTest, FiniteAndNonNegative) {
  const auto& s = starSamples();
  for (int i = 0; i < kStarSamples; ++i) {
    const Vec3 c = s.colors[i];
    ASSERT_TRUE(std::isfinite(c.x) && std::isfinite(c.y) && std::isfinite(c.z))
        << i;
    ASSERT_GE(c.x, 0.0f) << i;
    ASSERT_GE(c.y, 0.0f) << i;
    ASSERT_GE(c.z, 0.0f) << i;
  }
}

TEST(StarfieldTest, DeterministicBitwise) {
  const auto& s = starSamples();
  for (int i = 0; i < 5000; ++i) {
    const Vec3 again = sky::starfield(s.dirs[i]);
    ASSERT_EQ(std::memcmp(&again, &s.colors[i], sizeof(Vec3)), 0) << i;
  }
}

TEST(StarfieldTest, StarCoverageFractionIsPlausible) {
  // Statistical contract: bright (lum > 0.3) samples are a small but
  // non-negligible part of the sky: 0.1% .. 5%.
  const auto& s = starSamples();
  int bright = 0;
  for (const Vec3& c : s.colors) bright += luminance(c) > 0.3;
  const double frac = static_cast<double>(bright) / kStarSamples;
  EXPECT_GE(frac, 0.001) << "too few stars: " << frac;
  EXPECT_LE(frac, 0.05) << "too many stars: " << frac;
}

TEST(StarfieldTest, BackgroundIsDarkAndStarsReachFullBrightness) {
  const auto& s = starSamples();
  std::vector<double> lum;
  lum.reserve(s.colors.size());
  for (const Vec3& c : s.colors) lum.push_back(luminance(c));
  std::nth_element(lum.begin(), lum.begin() + lum.size() / 2, lum.end());
  EXPECT_LT(lum[lum.size() / 2], 0.05) << "median luminance";
  EXPECT_GE(*std::max_element(lum.begin(), lum.end()), 1.0) << "max luminance";
}

TEST(StarfieldTest, StarsAreNotMonochrome) {
  // Temperature tint: bright stars must differ in hue (b/r ratio spread).
  const auto& s = starSamples();
  double lo = 1e30, hi = -1e30;
  for (const Vec3& c : s.colors) {
    if (luminance(c) <= 0.3 || c.x <= 0) continue;
    const double ratio = c.z / c.x;
    lo = std::min(lo, ratio);
    hi = std::max(hi, ratio);
  }
  ASSERT_LT(lo, hi) << "no bright stars found";
  EXPECT_GT(hi / lo, 1.1) << "stars all share one color";
}

TEST(StarfieldTest, TinyPerturbationBarelyChangesColor) {
  // Pure function of direction => continuous: a 1e-5 rad nudge is far below
  // the smallest allowed star radius (~0.1 deg = 1.7e-3 rad). For a Gaussian
  // blob of width sigma and peak P, |dL| <= P*0.61*1e-5/sigma (~0.05 even for
  // sigma = 6e-4 rad, P = 5), so the bound is 0.05 + 10% of the luminance.
  const auto& s = starSamples();
  Rng rng(77);
  int total = 0, close = 0;
  auto check = [&](Vec3 d, Vec3 c) {
    const double l = luminance(c);
    const double l2 = luminance(sky::starfield(perturbed(d, 1e-5, rng)));
    ++total;
    close += std::fabs(l2 - l) <= 0.05 + 0.1 * l;
  };
  for (int i = 0; i < 20000; ++i) check(s.dirs[i], s.colors[i]);
  int stars = 0;  // also hit the interesting (star) samples specifically
  for (int i = 0; i < kStarSamples && stars < 2000; ++i) {
    if (luminance(s.colors[i]) > 0.3) {
      check(s.dirs[i], s.colors[i]);
      ++stars;
    }
  }
  // The 0.1% coverage floor guarantees >= 200 bright samples.
  ASSERT_GE(stars, 50) << "not enough star samples to be meaningful";
  EXPECT_GE(static_cast<double>(close) / total, 0.99);
}

TEST(StarfieldTest, LargePerturbationDecorrelates) {
  // A star is ~0.2 deg across; 5 deg away it must be gone for (nearly) every
  // star, and the field must not be constant.
  const auto& s = starSamples();
  Rng rng(78);
  int stars = 0, changed = 0;
  for (int i = 0; i < kStarSamples && stars < 2000; ++i) {
    if (luminance(s.colors[i]) <= 0.3) continue;
    ++stars;
    const Vec3 c2 = sky::starfield(perturbed(s.dirs[i], 5.0 * kPi / 180, rng));
    changed += std::fabs(luminance(c2) - luminance(s.colors[i])) > 0.1;
  }
  ASSERT_GE(stars, 50);
  EXPECT_GE(static_cast<double>(changed) / stars, 0.9);
}

TEST(StarfieldTest, NoSeamArtifactAtEquirectWrap) {
  // Mean luminance in a +-3 deg longitude band around u = 0 (the -z seam)
  // vs around u = 0.5 (+z), uniform-on-sphere within each band. The wide band
  // and 400k samples keep enough stars in each band (even at the 0.1%
  // coverage floor) for a 3x ratio window to be generous; a seam bug (empty
  // or doubled band) falls far outside it.
  auto bandMean = [](double lonCenter) {
    Rng rng(999);
    const double halfWidth = 3.0 * kPi / 180;
    double sum = 0;
    const int n = 400000;
    for (int i = 0; i < n; ++i) {
      const double lon = lonCenter + (2 * rng.uniform() - 1) * halfWidth;
      const double y = 2.0 * rng.uniform() - 1.0;
      const double s = std::sqrt(1.0 - y * y);
      sum += luminance(sky::starfield({static_cast<float>(s * std::sin(lon)),
                                       static_cast<float>(y),
                                       static_cast<float>(s * std::cos(lon))}));
    }
    return sum / n;
  };
  const double seam = bandMean(kPi);   // lon = pi -> direction -z -> u = 0 / 1
  const double front = bandMean(0.0);  // direction +z -> u = 0.5
  ASSERT_GT(seam, 0.0);
  ASSERT_GT(front, 0.0);
  EXPECT_LT(seam / front, 3.0);
  EXPECT_GT(seam / front, 1.0 / 3.0);
}

TEST(StarfieldTest, HemisphereBalance) {
  const auto& s = starSamples();
  int up = 0, down = 0;
  for (int i = 0; i < kStarSamples; ++i) {
    if (luminance(s.colors[i]) <= 0.3) continue;
    (s.dirs[i].y > 0 ? up : down)++;
  }
  const int n = up + down;
  ASSERT_GE(n, 50);
  // Spec: within +-30% (ratio 0.7..1.3, i.e. up-fraction 0.41..0.565, half
  // width 0.075). At the 0.1% floor n is only ~200 and bright samples cluster
  // (several samples per star, ~8 assumed), so widen by 4 sigma of the
  // clustered binomial: 4 * 0.5 * sqrt(8 / n). With thousands of samples this
  // is ~0.08; a one-hemisphere bug (fraction 1.0) is still caught for n >= 50.
  const double frac = static_cast<double>(up) / n;
  EXPECT_LE(std::fabs(frac - 0.5), 0.075 + 4 * 0.5 * std::sqrt(8.0 / n))
      << "up=" << up << " down=" << down;
}

// ---------------------------------------------------------------------------
// Image loading
// ---------------------------------------------------------------------------

std::string tempPath(const std::string& name) {
  return testing::TempDir() + "blackhole_sky_" + name;
}

double srgbToLinear(double c) {  // exact IEC 61966-2-1 EOTF
  return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
}

TEST(SkyImageLoadTest, PngIsDecodedWithExactSrgbEotfAndTopRowFirst) {
  // 2x2: row 0 = (0,128,255) (255,0,128); row 1 = (128,255,0) (64,64,64).
  const std::uint8_t px[] = {0, 128, 255, 255, 0, 128, 128, 255, 0, 64, 64, 64};
  const std::string path = tempPath("srgb.png");
  ASSERT_NE(stbi_write_png(path.c_str(), 2, 2, 3, px, 2 * 3), 0);

  const SkyImage img = sky::loadSkyImage(path);
  ASSERT_EQ(img.width, 2);
  ASSERT_EQ(img.height, 2);
  ASSERT_EQ(img.rgb.size(), 12u);
  // 1e-4: float pow/rounding is ~1e-7; skipping the EOTF is off by 0.29 at
  // 128, so the tolerance is tight enough to be meaningful.
  for (size_t i = 0; i < 12; ++i) {
    EXPECT_NEAR(img.rgb[i], srgbToLinear(px[i] / 255.0), 1e-4) << i;
  }
  EXPECT_NEAR(img.rgb[0], 0.0f, 1e-4);        // 0 -> 0
  EXPECT_NEAR(img.rgb[2], 1.0f, 1e-4);        // 255 -> 1
  EXPECT_NEAR(img.rgb[1], 0.2158605f, 1e-4);  // 128 -> 0.21586 (not 0.502)
  std::remove(path.c_str());
}

TEST(SkyImageLoadTest, PngAlphaIsDroppedToThreeChannels) {
  const std::uint8_t px[] = {255, 0, 0, 77, 0, 255, 0, 200};  // 2x1 RGBA
  const std::string path = tempPath("rgba.png");
  ASSERT_NE(stbi_write_png(path.c_str(), 2, 1, 4, px, 2 * 4), 0);
  const SkyImage img = sky::loadSkyImage(path);
  ASSERT_EQ(img.width, 2);
  ASSERT_EQ(img.height, 1);
  ASSERT_EQ(img.rgb.size(), 6u);
  const float want[] = {1, 0, 0, 0, 1, 0};
  for (int i = 0; i < 6; ++i) EXPECT_NEAR(img.rgb[i], want[i], 1e-4) << i;
  std::remove(path.c_str());
}

TEST(SkyImageLoadTest, HdrIsLoadedAsFloatWithoutGamma) {
  // Radiance RGBE is exact for these power-of-two triples. Values > 1 must not
  // be clamped and nothing may be run through the sRGB curve (0.5 stays 0.5).
  const float px[] = {0.5f,  1.0f,   2.0f,   // (0,0)
                      4.0f,  4.0f,   4.0f,   // (1,0)
                      0.25f, 0.125f, 0.25f,  // (0,1)
                      8.0f,  0.5f,   1.0f};  // (1,1)
  const std::string path = tempPath("hdr.hdr");
  ASSERT_NE(stbi_write_hdr(path.c_str(), 2, 2, 3, px), 0);
  const SkyImage img = sky::loadSkyImage(path);
  ASSERT_EQ(img.width, 2);
  ASSERT_EQ(img.height, 2);
  ASSERT_EQ(img.rgb.size(), 12u);
  for (size_t i = 0; i < 12; ++i) {
    EXPECT_NEAR(img.rgb[i], px[i], 0.01f * px[i]) << i;  // RGBE: 8-bit mantissa
  }
  std::remove(path.c_str());
}

TEST(SkyImageLoadTest, GrayscalePngIsExpandedToRgb) {
  const std::uint8_t px[] = {0, 128, 255};  // 3x1, 1 channel
  const std::string path = tempPath("gray.png");
  ASSERT_NE(stbi_write_png(path.c_str(), 3, 1, 1, px, 3), 0);
  const SkyImage img = sky::loadSkyImage(path);
  ASSERT_EQ(img.width, 3);
  ASSERT_EQ(img.height, 1);
  ASSERT_EQ(img.rgb.size(), 9u);
  for (int i = 0; i < 3; ++i) {
    const double want = srgbToLinear(px[i] / 255.0);
    for (int c = 0; c < 3; ++c) {
      EXPECT_NEAR(img.rgb[i * 3 + c], want, 1e-4) << i << "," << c;
    }
  }
  std::remove(path.c_str());
}

TEST(SkyImageLoadTest, JpgIsDecodedAsSrgb) {
  // Solid 16x16 colour at quality 100: JPEG error on a flat block is a few
  // 8-bit levels at most; 0.02 in linear is >> that near 200/100/50 yet far
  // below the ~0.25 error of skipping the sRGB curve at 128-ish values.
  const int w = 16, h = 16;
  std::vector<std::uint8_t> px(w * h * 3);
  for (int i = 0; i < w * h; ++i) {
    px[i * 3 + 0] = 200;
    px[i * 3 + 1] = 100;
    px[i * 3 + 2] = 50;
  }
  const std::string path = tempPath("solid.jpg");
  ASSERT_NE(stbi_write_jpg(path.c_str(), w, h, 3, px.data(), 100), 0);
  const SkyImage img = sky::loadSkyImage(path);
  ASSERT_EQ(img.width, w);
  ASSERT_EQ(img.height, h);
  ASSERT_EQ(img.rgb.size(), static_cast<size_t>(w * h * 3));
  const double want[3] = {srgbToLinear(200 / 255.0), srgbToLinear(100 / 255.0),
                          srgbToLinear(50 / 255.0)};
  for (int c = 0; c < 3; ++c) {
    EXPECT_NEAR(img.rgb[8 * w * 3 + 8 * 3 + c], want[c], 0.02) << c;
  }
  std::remove(path.c_str());
}

TEST(SkyImageLoadTest, MissingFileThrowsWithPathInMessage) {
  const std::string path = tempPath("does_not_exist.png");
  try {
    sky::loadSkyImage(path);
    FAIL() << "expected std::runtime_error";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find(path), std::string::npos) << e.what();
  }
}

TEST(SkyImageLoadTest, GarbageFileThrowsWithPathInMessage) {
  for (const char* name : {"garbage.png", "garbage.hdr", "garbage.jpg"}) {
    const std::string path = tempPath(name);
    {
      std::ofstream f(path, std::ios::binary);
      f << "this is definitely not an image file\n";
    }
    try {
      sky::loadSkyImage(path);
      ADD_FAILURE() << "expected std::runtime_error for " << path;
    } catch (const std::runtime_error& e) {
      EXPECT_NE(std::string(e.what()).find(path), std::string::npos)
          << e.what();
    }
    std::remove(path.c_str());
  }
}

TEST(SkyImageLoadTest, SyntheticSkyMatchesDocumentedFunction) {
  const SkyImage img = sky::makeSyntheticSky(8, 4);
  ASSERT_EQ(img.width, 8);
  ASSERT_EQ(img.height, 4);
  ASSERT_EQ(img.rgb.size(), 8u * 4 * 3);
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 8; ++x) {
      const float* p = &img.rgb[(y * 8 + x) * 3];
      EXPECT_NEAR(p[0], x / 7.0f, 1e-6f);
      EXPECT_NEAR(p[1], y / 3.0f, 1e-6f);
      EXPECT_NEAR(p[2], 0.25f, 1e-6f);
    }
  }
}

// ---------------------------------------------------------------------------
// CPU bilinear reference sampler
// ---------------------------------------------------------------------------

// 4x2 image: r = (x + 4y)/8, g = (x+1)(y+1)/8, b = (7 - x - 3y)/16.
SkyImage image4x2() {
  SkyImage img;
  img.width = 4;
  img.height = 2;
  for (int y = 0; y < 2; ++y) {
    for (int x = 0; x < 4; ++x) {
      img.rgb.push_back((x + 4 * y) / 8.0f);
      img.rgb.push_back((x + 1) * (y + 1) / 8.0f);
      img.rgb.push_back((7 - x - 3 * y) / 16.0f);
    }
  }
  return img;
}

// Hand-rolled bilinear blend of texels (x0,y0),(x1,y0),(x0,y1),(x1,y1) with
// weights fx (towards x1) and fy (towards y1); channel c.
double blend(const SkyImage& img, int x0, int x1, double fx, int y0, int y1,
             double fy, int c) {
  auto t = [&](int x, int y) {
    return double(img.rgb[(y * img.width + x) * 3 + c]);
  };
  return (1 - fy) * ((1 - fx) * t(x0, y0) + fx * t(x1, y0)) +
         fy * ((1 - fx) * t(x0, y1) + fx * t(x1, y1));
}

void expectColor(Vec3 got, const SkyImage& img, int x0, int x1, double fx,
                 int y0, int y1, double fy, double tol = 1e-5) {
  EXPECT_NEAR(got.x, blend(img, x0, x1, fx, y0, y1, fy, 0), tol);
  EXPECT_NEAR(got.y, blend(img, x0, x1, fx, y0, y1, fy, 1), tol);
  EXPECT_NEAR(got.z, blend(img, x0, x1, fx, y0, y1, fy, 2), tol);
}

TEST(SampleSkyTest, TexelCentersReturnExactTexel) {
  // Catches texel-corner (missing -0.5) and v-flip mutants: both move every
  // sample by >= half a texel, i.e. >= 0.125 in r.
  const SkyImage img = image4x2();
  for (int y = 0; y < 2; ++y) {
    for (int x = 0; x < 4; ++x) {
      const Vec3 c =
          sky::sampleSky(img, dirFromUV((x + 0.5) / 4, (y + 0.5) / 2));
      expectColor(c, img, x, x, 0, y, y, 0);
    }
  }
}

TEST(SampleSkyTest, MidpointIsAverageOfFourTexels) {
  const SkyImage img = image4x2();
  // u = 0.5 is halfway between column centers 0.375 and 0.625; v = 0.5 is
  // halfway between row centers 0.25 and 0.75. By hand for channel r:
  // (1/8 + 2/8 + 5/8 + 6/8) / 4 = 14/32 = 0.4375. Nearest filtering fails.
  const Vec3 c = sky::sampleSky(img, dirFromUV(0.5, 0.5));
  EXPECT_NEAR(c.x, 0.4375f, 1e-5f);
  expectColor(c, img, 1, 2, 0.5, 0, 1, 0.5);
}

TEST(SampleSkyTest, UnequalWeightsAreExact) {
  const SkyImage img = image4x2();
  // u = 0.4375 -> x = 0.4375*4 - 0.5 = 1.25 (cols 1,2; fx = 0.25)
  // v = 0.375  -> y = 0.375*2 - 0.5 = 0.25 (rows 0,1; fy = 0.25)
  // r by hand: row0 = 1/8 + .25*(1/8) = 0.15625; row1 = 5/8 + .25/8 = 0.65625
  //            -> 0.75*0.15625 + 0.25*0.65625 = 0.28125
  const Vec3 c = sky::sampleSky(img, dirFromUV(0.4375, 0.375));
  EXPECT_NEAR(c.x, 0.28125f, 1e-5f);
  expectColor(c, img, 1, 2, 0.25, 0, 1, 0.25);
}

TEST(SampleSkyTest, WrapsAroundInU) {
  const SkyImage img = image4x2();
  // u = 0.9375 -> x = 3.25: blend of column 3 and (wrapped) column 0, fx=.25.
  expectColor(sky::sampleSky(img, dirFromUV(0.9375, 0.25)), img, 3, 0, 0.25, 0,
              0, 0);
  // u = 0.0625 -> x = -0.25: columns 3 (wrapped) and 0 with fx = 0.75.
  expectColor(sky::sampleSky(img, dirFromUV(0.0625, 0.25)), img, 3, 0, 0.75, 0,
              0, 0);
  // Just below 1: x ~ 3.5 -> nearly even mix of column 3 and column 0 (a
  // clamp-in-u implementation would return column 3 alone).
  const Vec3 c = sky::sampleSky(img, dirFromUV(0.99999, 0.25));
  EXPECT_NEAR(c.x, blend(img, 3, 0, 0.49996, 0, 0, 0, 0), 1e-3);
  EXPECT_NEAR(c.x, 0.5 * (3.0 / 8 + 0.0), 1e-3);
}

TEST(SampleSkyTest, ClampsInV) {
  const SkyImage img = image4x2();
  // v = 0.05 is above the first row center (0.25): pure row 0, no wrap to the
  // bottom row. Same at the bottom with v = 0.95 -> pure row 1.
  expectColor(sky::sampleSky(img, dirFromUV(0.375, 0.05)), img, 1, 1, 0, 0, 0,
              0);
  expectColor(sky::sampleSky(img, dirFromUV(0.375, 0.95)), img, 1, 1, 0, 1, 1,
              0);
  // Zenith / nadir directions: u = 0.5 (atan2(0,0) = 0) -> columns 1 and 2.
  expectColor(sky::sampleSky(img, {0, 1, 0}), img, 1, 2, 0.5, 0, 0, 0);
  expectColor(sky::sampleSky(img, {0, -1, 0}), img, 1, 2, 0.5, 1, 1, 0);
}

}  // namespace
