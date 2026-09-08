// Phase-3b tests: render() background modes. Escaped rays are coloured by the
// FINAL direction d = normalize(final velocity):
//   Gradient  -> cam::skyGradient(d)                     (unchanged)
//   Starfield -> sky::starfield(d)            (linear RGB, no tone mapping)
//   Image     -> sky::sampleSky(loadSkyImage(path), d)   (GPU: SkyTexture +
//                sampleSkyDevice)
// Horizon / MaxSteps -> (0,0,0). Bytes = lround(255 * clamp01(c)), alpha 255,
// i.e. no gamma / tone mapping in this phase (sky colours are linear values
// written straight to the byte buffer).
// Image mode throws std::runtime_error (message contains the path) if the file
// cannot be loaded; Gradient/Starfield never look at skyImagePath.
//
// CPU reference = the same camera.hpp / geodesic.hpp calls and
// renderTraceParams as render_shadow_test.cpp. Host vs device builds differ by
// a few ulp (FMA contraction, sinf/atan2f), which can move a final direction
// by ~1e-5 and flip the outcome of rays within ~1e-4 of the critical impact
// parameter, hence the percentage-based tolerances below.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "blackhole/camera.hpp"
#include "blackhole/geodesic.hpp"
#include "blackhole/image.hpp"
#include "blackhole/render.hpp"
#include "blackhole/sky.hpp"
#include "cuda_test_support.hpp"
#include "stb_image_write.h"  // implementation lives in src/image.cpp

namespace {

using blackhole::Image;
using blackhole::RenderConfig;
using blackhole::SkyMode;
namespace cam = blackhole::cam;
namespace geo = blackhole::geo;
namespace sky = blackhole::sky;

constexpr double kPi = 3.14159265358979323846;
constexpr double kBc = 2.598076211353316;  // 1.5 * sqrt(3)

RenderConfig makeConfig(int w, int h, float fov, float el, float dist,
                        SkyMode mode = SkyMode::Gradient,
                        const std::string& path = "") {
  RenderConfig c;
  c.width = w;
  c.height = h;
  c.fovDegrees = fov;
  c.elevationDegrees = el;
  c.observerDistance = dist;
  c.sky = mode;
  c.skyImagePath = path;
  return c;
}

std::string tempPath(const std::string& name) {
  return testing::TempDir() + "blackhole_render_sky_" + name;
}

// Removes the file on scope exit so early ASSERTs do not leak fixtures.
struct TempFile {
  std::string path;
  explicit TempFile(const std::string& name) : path(tempPath(name)) {
    std::remove(path.c_str());
  }
  ~TempFile() { std::remove(path.c_str()); }
};

std::uint8_t quantize(float c) {
  const float cl = c < 0.0f ? 0.0f : (c > 1.0f ? 1.0f : c);
  return static_cast<std::uint8_t>(std::lround(255.0f * cl));
}

int channel(const Image& im, int x, int y, int ch) {
  return im.data()[im.pixelIndex(x, y) + ch];
}

bool isBlack(const Image& im, int x, int y) {
  return channel(im, x, y, 0) == 0 && channel(im, x, y, 1) == 0 &&
         channel(im, x, y, 2) == 0;
}

bool sameBytes(const Image& a, const Image& b) {
  return a.size() == b.size() && std::equal(a.data(), a.data() + a.size(),
                                            b.data());
}

geo::Vec3 toGeo(cam::Vec3 v) { return {v.x, v.y, v.z}; }

cam::Camera cameraOf(const RenderConfig& c) {
  return cam::makeCamera(c.width, c.height, c.observerDistance, c.fovDegrees,
                         c.elevationDegrees);
}

// CPU trace of one pixel: outcome plus the unit final direction.
struct Traced {
  bool escaped = false;
  sky::Vec3 dir{0, 0, 0};
};

Traced cpuTrace(const RenderConfig& config, const cam::Camera& camera, int x,
                int y) {
  const geo::Ray ray = geo::makeRay(
      toGeo(camera.pos), toGeo(cam::primaryRayDir(camera, x, y)));
  const geo::TraceResult res =
      geo::trace(ray, blackhole::renderTraceParams(config));
  Traced t;
  if (res.outcome == geo::Outcome::Escaped) {
    const geo::Vec3 d = geo::normalize(res.ray.vel);
    t.escaped = true;
    t.dir = {d.x, d.y, d.z};
  }
  return t;
}

struct Rgb8 {
  std::uint8_t c[3];
};

// Expected pixel for sky function `colorOf(dir) -> sky::Vec3`.
template <typename F>
std::vector<Rgb8> cpuReference(const RenderConfig& config, F colorOf,
                               std::vector<Traced>* traced = nullptr) {
  const cam::Camera camera = cameraOf(config);
  std::vector<Rgb8> out(static_cast<std::size_t>(config.width) * config.height);
  if (traced) traced->resize(out.size());
  for (int y = 0; y < config.height; ++y) {
    for (int x = 0; x < config.width; ++x) {
      const Traced t = cpuTrace(config, camera, x, y);
      Rgb8 p{{0, 0, 0}};
      if (t.escaped) {
        const sky::Vec3 c = colorOf(t.dir);
        p = Rgb8{{quantize(c.x), quantize(c.y), quantize(c.z)}};
      }
      const std::size_t i = static_cast<std::size_t>(y) * config.width + x;
      out[i] = p;
      if (traced) (*traced)[i] = t;
    }
  }
  return out;
}

struct Agreement {
  int channelsOff = 0;   // |gpu - cpu| > tol
  int channelsBig = 0;   // |gpu - cpu| > 8
  int classDiff = 0;     // black vs non-black differs
  int total = 0;
};

Agreement compare(const Image& gpu, const std::vector<Rgb8>& cpu, int tol) {
  Agreement a;
  a.total = gpu.width() * gpu.height();
  for (int y = 0; y < gpu.height(); ++y) {
    for (int x = 0; x < gpu.width(); ++x) {
      const Rgb8& e = cpu[static_cast<std::size_t>(y) * gpu.width() + x];
      bool cb = true, gb = true;
      bool bigPixel = false;
      for (int ch = 0; ch < 3; ++ch) {
        const int g = channel(gpu, x, y, ch);
        const int d = std::abs(g - int{e.c[ch]});
        if (d > tol) ++a.channelsOff;
        bigPixel = bigPixel || d > 8;
        cb = cb && e.c[ch] == 0;
        gb = gb && g == 0;
      }
      a.channelsBig += bigPixel;
      a.classDiff += cb != gb;
    }
  }
  return a;
}

double luminance(const Image& im, int x, int y) {
  return (0.2126 * channel(im, x, y, 0) + 0.7152 * channel(im, x, y, 1) +
          0.0722 * channel(im, x, y, 2)) /
         255.0;
}

double srgbEncode(double c) {  // exact IEC 61966-2-1 OETF
  return c <= 0.0031308 ? 12.92 * c : 1.055 * std::pow(c, 1.0 / 2.4) - 0.055;
}

// Quantise a linear float sky to an 8-bit sRGB PNG.
void writeSkyPng(const std::string& path, const sky::SkyImage& img) {
  std::vector<std::uint8_t> px(img.rgb.size());
  for (std::size_t i = 0; i < px.size(); ++i) {
    const double c = std::min(1.0, std::max(0.0, double{img.rgb[i]}));
    px[i] = static_cast<std::uint8_t>(std::lround(255.0 * srgbEncode(c)));
  }
  ASSERT_NE(stbi_write_png(path.c_str(), img.width, img.height, 3, px.data(),
                           img.width * 3),
            0)
      << path;
}

// Left half (u < 0.5) pure red, right half pure blue, as an sRGB PNG
// (255 -> linear 1.0 exactly, 0 -> 0).
void writeRedBluePng(const std::string& path) {
  const int w = 64, h = 32;
  std::vector<std::uint8_t> px(static_cast<std::size_t>(w) * h * 3, 0);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      std::uint8_t* p = &px[(static_cast<std::size_t>(y) * w + x) * 3];
      if (x < w / 2) p[0] = 255; else p[2] = 255;
    }
  }
  ASSERT_NE(stbi_write_png(path.c_str(), w, h, 3, px.data(), w * 3), 0);
}

// ---------------------------------------------------------------------------
// 1. Starfield: CPU/GPU agreement
// ---------------------------------------------------------------------------

struct SkyCase {
  int w, h;
  float fov, el, dist;
};

class RenderStarfieldAgreementTest : public ::testing::TestWithParam<SkyCase> {
};

// Tolerances: the starfield is a sum of Gaussians with sigma ~0.1-0.25 deg and
// peak up to ~3.4 (bytes saturate), so near a star edge a 1e-5 direction
// difference between host and device builds moves a byte by a few units; and
// rays within ~1e-4 of b_c can flip outcome. We require >= 99% of channels
// within +-2 bytes, <= 0.5% of pixels with a differing black/non-black
// classification, and <= 1% of pixels with any channel off by more than 8.
// Shadow pixels (Horizon/MaxSteps on the CPU) must be exactly (0,0,0) on the
// GPU apart from those same flips (counted in the classification budget).
TEST_P(RenderStarfieldAgreementTest, MatchesCpuReference) {
  SKIP_IF_NO_CUDA_DEVICE();
  const SkyCase p = GetParam();
  const RenderConfig config =
      makeConfig(p.w, p.h, p.fov, p.el, p.dist, SkyMode::Starfield);
  std::vector<Traced> traced;
  const std::vector<Rgb8> cpu = cpuReference(
      config, [](sky::Vec3 d) { return sky::starfield(d); }, &traced);
  const Image image = blackhole::render(config);
  ASSERT_EQ(image.width(), p.w);
  ASSERT_EQ(image.height(), p.h);

  int shadowLit = 0, shadow = 0, badAlpha = 0;
  for (int y = 0; y < p.h; ++y) {
    for (int x = 0; x < p.w; ++x) {
      badAlpha += channel(image, x, y, 3) != 255;
      if (!traced[static_cast<std::size_t>(y) * p.w + x].escaped) {
        ++shadow;
        shadowLit += !isBlack(image, x, y);
      }
    }
  }
  const Agreement a = compare(image, cpu, 2);
  EXPECT_EQ(badAlpha, 0);
  EXPECT_GT(shadow, 0);  // non-vacuity: the shadow is in view
  EXPECT_LE(shadowLit, a.total * 5 / 1000);
  EXPECT_LE(a.channelsOff, a.total * 3 / 100);
  EXPECT_LE(a.classDiff, a.total * 5 / 1000);
  EXPECT_LE(a.channelsBig, a.total / 100);
}

INSTANTIATE_TEST_SUITE_P(
    Starfield, RenderStarfieldAgreementTest,
    ::testing::Values(SkyCase{64, 48, 60.0f, 0.0f, 20.0f},
                      SkyCase{64, 48, 60.0f, 30.0f, 20.0f},
                      SkyCase{64, 48, 90.0f, 0.0f, 10.0f},
                      SkyCase{64, 48, 90.0f, 30.0f, 10.0f},
                      // Large enough that individual stars land on pixels.
                      SkyCase{192, 108, 90.0f, 0.0f, 20.0f}));

// ---------------------------------------------------------------------------
// 2. The starfield render looks like a starfield
// ---------------------------------------------------------------------------

// ~0.12 stars per square degree vs 0.39 deg^2 per pixel here: a small
// percentage of pixels is lit by a star (> 0.3 luminance), the rest is the
// near-black sky plus the faint (<= 0.012) galactic band (<= 3 bytes).
TEST(RenderStarfieldTest, LooksLikeStarsOnBlack) {
  SKIP_IF_NO_CUDA_DEVICE();
  const RenderConfig config =
      makeConfig(256, 144, 90.0f, 0.0f, 20.0f, SkyMode::Starfield);
  const Image stars = blackhole::render(config);
  const Image grad = blackhole::render(makeConfig(256, 144, 90.0f, 0.0f, 20.0f));

  std::vector<double> lum;
  int bright = 0;
  long long diffSum = 0;
  for (int y = 0; y < 144; ++y) {
    for (int x = 0; x < 256; ++x) {
      for (int ch = 0; ch < 3; ++ch)
        diffSum += std::abs(channel(stars, x, y, ch) - channel(grad, x, y, ch));
      if (isBlack(grad, x, y)) {
        EXPECT_TRUE(isBlack(stars, x, y)) << x << "," << y;  // shadow
        continue;
      }
      const double l = luminance(stars, x, y);
      lum.push_back(l);
      bright += l > 0.3;
    }
  }
  ASSERT_GT(lum.size(), 20000u);
  const double frac = static_cast<double>(bright) / lum.size();
  EXPECT_GE(frac, 0.0005) << "bright fraction";
  EXPECT_LE(frac, 0.08) << "bright fraction";
  std::nth_element(lum.begin(), lum.begin() + lum.size() / 2, lum.end());
  EXPECT_LT(lum[lum.size() / 2], 0.05) << "median luminance";
  // Gradient pixels average ~128/channel, the starfield ~0: mean |diff| >> 50.
  EXPECT_GT(static_cast<double>(diffSum) / (256.0 * 144.0 * 3.0), 50.0);
}

// ---------------------------------------------------------------------------
// 3. Image mode vs CPU reference
// ---------------------------------------------------------------------------

// The CPU reference loads the same PNG (sky::loadSkyImage) and bilinearly
// samples it (sky::sampleSky) with the CPU final direction; the GPU uses a
// float4 texture whose hardware filter weights have 8 fractional bits (error
// <= 1/512 of the texel difference, here a few thousandths of 1.0 = <= 1
// byte) plus the host/device direction noise (1e-5 of a texel step). So we
// require >= 99% of channels within +-3 bytes, <= 0.5% black/non-black
// classification flips (shadow edge) and <= 1% of pixels with any channel off
// by more than 8 (pixels whose direction straddles the u=0/1 seam of the
// synthetic gradient, where the sky jumps from 1 to 0).
struct SkySize {
  int w, h;
};

class RenderImageSkyAgreementTest : public ::testing::TestWithParam<SkySize> {
};

// Coarse odd-sized skies (7x3, 15x7) make a half-texel shift or a wrong
// v-scale (e.g. dividing by h-1) a many-percent colour error instead of a
// sub-byte one, so the same tolerances catch them.
TEST_P(RenderImageSkyAgreementTest, MatchesCpuReference) {
  SKIP_IF_NO_CUDA_DEVICE();
  const SkySize sz = GetParam();
  TempFile file("synthetic_" + std::to_string(sz.w) + "x" +
                std::to_string(sz.h) + ".png");
  writeSkyPng(file.path, sky::makeSyntheticSky(sz.w, sz.h));
  const sky::SkyImage image = sky::loadSkyImage(file.path);

  for (const float el : {0.0f, 30.0f}) {
    const RenderConfig config =
        makeConfig(64, 48, 60.0f, el, 20.0f, SkyMode::Image, file.path);
    const std::vector<Rgb8> cpu = cpuReference(
        config, [&](sky::Vec3 d) { return sky::sampleSky(image, d); });
    const Image gpu = blackhole::render(config);
    ASSERT_EQ(gpu.width(), 64);
    ASSERT_EQ(gpu.height(), 48);
    const Agreement a = compare(gpu, cpu, 3);
    EXPECT_LE(a.channelsOff, a.total * 3 / 100) << "el " << el;
    EXPECT_LE(a.classDiff, a.total * 5 / 1000) << "el " << el;
    EXPECT_LE(a.channelsBig, a.total / 100) << "el " << el;
    // Non-vacuity: image colours are visible (not all black) and alpha 255.
    // The synthetic sky has b = 0.25 linear everywhere (and PNG quantisation
    // round-trips it to 64 +- 1), so every lit pixel has blue ~ 64.
    int lit = 0, badBlue = 0;
    for (int y = 0; y < 48; ++y)
      for (int x = 0; x < 64; ++x) {
        ASSERT_EQ(channel(gpu, x, y, 3), 255);
        if (!isBlack(gpu, x, y)) {
          ++lit;
          badBlue += std::abs(channel(gpu, x, y, 2) - 64) > 3;
        }
      }
    EXPECT_GT(lit, 48 * 64 / 2) << "el " << el;
    EXPECT_LE(badBlue, a.total / 100) << "el " << el;
    EXPECT_TRUE(isBlack(gpu, 32, 24));  // centre ray is in the shadow
  }
}

INSTANTIATE_TEST_SUITE_P(Image, RenderImageSkyAgreementTest,
                         ::testing::Values(SkySize{64, 32}, SkySize{7, 3},
                                           SkySize{15, 7}));

TEST(RenderImageSkyTest, DifferentImageGivesDifferentRender) {
  SKIP_IF_NO_CUDA_DEVICE();
  TempFile tf_a("img_a.png");
  const std::string& a = tf_a.path;
  TempFile tf_b("img_b.png");
  const std::string& b = tf_b.path;
  sky::SkyImage ia = sky::makeSyntheticSky(64, 32);
  sky::SkyImage ib = ia;  // swap red and green: a visibly different sky
  for (std::size_t i = 0; i + 2 < ib.rgb.size(); i += 3)
    std::swap(ib.rgb[i], ib.rgb[i + 1]);
  writeSkyPng(a, ia);
  writeSkyPng(b, ib);
  const Image ra =
      blackhole::render(makeConfig(64, 48, 60.0f, 0.0f, 20.0f, SkyMode::Image, a));
  const Image rb =
      blackhole::render(makeConfig(64, 48, 60.0f, 0.0f, 20.0f, SkyMode::Image, b));
  long long diff = 0;
  for (std::size_t i = 0; i < ra.size(); ++i)
    diff += std::abs(int{ra.data()[i]} - int{rb.data()[i]});
  EXPECT_GT(diff / static_cast<double>(ra.size()), 5.0);
}

// ---------------------------------------------------------------------------
// 4. Lensing is visible in Image mode
// ---------------------------------------------------------------------------

// Camera at -z looking +z with right = +x, el 0. A red (u < 0.5) / blue
// (u >= 0.5) sky: the unlensed direction of a pixel on the +x side has x > 0
// and z > 0, i.e. u in (0.5, 0.75) -> blue (renderFlat shows blue with the
// gradient too; here we use the CPU primary direction). Just outside the
// shadow (1.15 and 1.3 shadow radii; b ~ 1.15-1.3 b_c) the ray is deflected by
// more than a radian towards -x, so the final direction has x < -0.2 (u < 0.5
// whichever side of z it lies: atan2(x<0, z) is in (-pi, 0)) -> red. The mirror
// pixel on the -x side turns blue. We assert the CPU direction really is on the
// other hemisphere (non-vacuity of the geometry), the GPU colour is the clearly
// red/blue colour, and that it is the OPPOSITE of the unlensed colour.
TEST(RenderImageSkyTest, LensingFlipsHemisphereJustOutsideTheShadow) {
  SKIP_IF_NO_CUDA_DEVICE();
  TempFile tf("redblue.png");
  const std::string& path = tf.path;
  writeRedBluePng(path);
  const RenderConfig config =
      makeConfig(400, 400, 60.0f, 0.0f, 20.0f, SkyMode::Image, path);
  const cam::Camera camera = cameraOf(config);
  const double alphaC = std::asin(kBc / config.observerDistance);
  const double rpx = std::tan(alphaC) /
                     std::tan(config.fovDegrees * kPi / 180.0 * 0.5) *
                     (config.height / 2.0);
  const Image lensed = blackhole::render(config);
  const sky::SkyImage rb = sky::loadSkyImage(path);

  for (const double mult : {1.15, 1.3}) {
    const int off = static_cast<int>(std::lround(mult * rpx));
    for (const int sign : {+1, -1}) {
      const int x = 200 + (sign > 0 ? off : -off - 1);
      const int y = 200;
      const cam::Vec3 prim = cam::primaryRayDir(camera, x, y);
      const Traced t = cpuTrace(config, camera, x, y);
      ASSERT_TRUE(t.escaped) << mult << " " << sign;
      // Primary on side `sign`, final direction clearly on the other side.
      ASSERT_GT(prim.x * sign, 0.05f);
      ASSERT_LT(t.dir.x * sign, -0.2f) << mult << " " << sign;
      // Expected colours (+x: unlensed blue, lensed red; -x: the reverse).
      const sky::Vec3 want = sky::sampleSky(rb, t.dir);
      const sky::Vec3 unlensed = sky::sampleSky(
          rb, sky::Vec3{prim.x, prim.y, prim.z});
      const int wantR = sign > 0 ? 255 : 0, wantB = sign > 0 ? 0 : 255;
      ASSERT_NEAR(want.x * 255.0f, wantR, 1.0f);  // CPU reference is pure
      ASSERT_NEAR(unlensed.z * 255.0f, wantR ? 255 : 0, 1.0f);
      EXPECT_NEAR(channel(lensed, x, y, 0), wantR, 3) << mult << " " << sign;
      EXPECT_NEAR(channel(lensed, x, y, 1), 0, 3) << mult << " " << sign;
      EXPECT_NEAR(channel(lensed, x, y, 2), wantB, 3) << mult << " " << sign;
      // ... and it differs from the colour of the unlensed (primary) ray.
      EXPECT_NEAR(channel(lensed, x, y, 0), quantize(want.x), 3);
      EXPECT_GT(std::abs(channel(lensed, x, y, 0) -
                         int{quantize(unlensed.x)}),
                200);
    }
  }
}

// ---------------------------------------------------------------------------
// 5. Errors, determinism, Gradient unchanged
// ---------------------------------------------------------------------------

TEST(RenderSkyErrorTest, MissingImageThrowsRuntimeErrorWithPath) {
  TempFile tf("does_not_exist.png");
  const std::string& path = tf.path;
  const RenderConfig config =
      makeConfig(16, 16, 60.0f, 0.0f, 20.0f, SkyMode::Image, path);
  try {
    blackhole::render(config);
    FAIL() << "expected std::runtime_error";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find(path), std::string::npos) << e.what();
  } catch (...) {
    FAIL() << "expected std::runtime_error";
  }
}

TEST(RenderSkyErrorTest, GarbageImageThrowsRuntimeErrorWithPath) {
  TempFile tf("garbage.png");
  const std::string& path = tf.path;
  {
    std::ofstream f(path, std::ios::binary);
    f << "this is definitely not an image file\n";
  }
  const RenderConfig config =
      makeConfig(16, 16, 60.0f, 0.0f, 20.0f, SkyMode::Image, path);
  try {
    blackhole::render(config);
    ADD_FAILURE() << "expected std::runtime_error";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find(path), std::string::npos) << e.what();
  } catch (...) {
    ADD_FAILURE() << "expected std::runtime_error";
  }
}

TEST(RenderSkyErrorTest, EmptyImagePathIsInvalidArgument) {
  EXPECT_THROW(blackhole::render(makeConfig(16, 16, 60.0f, 0.0f, 20.0f,
                                            SkyMode::Image, "")),
               std::invalid_argument);
}

TEST(RenderSkyTest, ImageModeTwiceInARowIsIdentical) {
  SKIP_IF_NO_CUDA_DEVICE();
  TempFile tf("twice.png");
  const std::string& path = tf.path;
  writeSkyPng(path, sky::makeSyntheticSky(64, 32));
  const RenderConfig config =
      makeConfig(64, 48, 60.0f, 30.0f, 20.0f, SkyMode::Image, path);
  const Image a = blackhole::render(config);
  const Image b = blackhole::render(config);
  EXPECT_TRUE(sameBytes(a, b));
  int lit = 0;
  for (int y = 0; y < 48; ++y)
    for (int x = 0; x < 64; ++x) lit += !isBlack(a, x, y);
  EXPECT_GT(lit, 48 * 64 / 2);  // not an all-black buffer
  // And it is not the gradient render (stale/ignored image state).
  EXPECT_FALSE(sameBytes(a, blackhole::render(makeConfig(64, 48, 60.0f, 30.0f, 20.0f))));
}

TEST(RenderSkyTest, GradientIgnoresImagePathAndMatchesDefault) {
  SKIP_IF_NO_CUDA_DEVICE();
  const Image def = blackhole::render(makeConfig(64, 48, 60.0f, 30.0f, 20.0f));
  // Gradient never touches the path (it need not exist, and must not throw).
  const Image grad = blackhole::render(
      makeConfig(64, 48, 60.0f, 30.0f, 20.0f, SkyMode::Gradient,
                 tempPath("never_read.png")));
  EXPECT_TRUE(sameBytes(def, grad));
  // And it is the gradient: top-left pixel is 0.5 + 0.5*d bytes, not starfield.
  const Image stars = blackhole::render(
      makeConfig(64, 48, 60.0f, 30.0f, 20.0f, SkyMode::Starfield));
  EXPECT_FALSE(sameBytes(def, stars));
}

TEST(RenderSkyTest, StarfieldIsDeterministicAcrossCalls) {
  SKIP_IF_NO_CUDA_DEVICE();
  const RenderConfig config =
      makeConfig(96, 64, 60.0f, 30.0f, 20.0f, SkyMode::Starfield);
  const Image a = blackhole::render(config);
  const Image b = blackhole::render(config);
  EXPECT_TRUE(sameBytes(a, b));
  EXPECT_FALSE(sameBytes(
      a, blackhole::render(makeConfig(96, 64, 60.0f, 30.0f, 20.0f))));
}

// Resource hygiene: repeated Image renders must release the cudaArray/texture
// (a 1024x512 float4 sky is 8 MB, so a leak over 10 renders is ~80 MB; we allow
// 64 MB of allocator noise), and a failed render must not leave a sticky CUDA
// error behind.
TEST(RenderSkyTest, NoLeakAndNoStickyErrorAcrossRenders) {
  SKIP_IF_NO_CUDA_DEVICE();
  TempFile file("big.png");
  writeSkyPng(file.path, sky::makeSyntheticSky(1024, 512));
  const RenderConfig config =
      makeConfig(32, 24, 60.0f, 30.0f, 20.0f, SkyMode::Image, file.path);
  (void)blackhole::render(config);  // warm-up: context, caches
  std::size_t freeBefore = 0, freeAfter = 0, total = 0;
  ASSERT_EQ(cudaMemGetInfo(&freeBefore, &total), cudaSuccess);
  for (int i = 0; i < 10; ++i) (void)blackhole::render(config);
  ASSERT_EQ(cudaMemGetInfo(&freeAfter, &total), cudaSuccess);
  const long long dropMB =
      (static_cast<long long>(freeBefore) - static_cast<long long>(freeAfter)) /
      (1024 * 1024);
  EXPECT_LE(dropMB, 64) << "free memory dropped by " << dropMB << " MB";

  TempFile missing("hygiene_missing.png");
  EXPECT_THROW(blackhole::render(makeConfig(16, 16, 60.0f, 0.0f, 20.0f,
                                            SkyMode::Image, missing.path)),
               std::runtime_error);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  const Image ok = blackhole::render(makeConfig(16, 16, 60.0f, 0.0f, 20.0f));
  EXPECT_EQ(ok.width(), 16);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

}  // namespace
