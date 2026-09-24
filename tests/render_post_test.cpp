// Phase-5b tests: post-processing inside render() -- supersampling, bloom and
// tone mapping (post.hpp) on top of the Phase-2 lensing trace.
//
// Pipeline semantics pinned here (per output pixel (px, py), N = spp):
//   sample i: o = post::sampleOffset(i, spp, px, py); fx = px + o.dx,
//     fy = py + o.dy; ray = geo::makeRay(cam.pos, cam::primaryRayDirSub(cam,
//     fx, fy)); traced with renderTraceParams(config); linear radiance L_i is
//     black for Horizon/MaxSteps, else cam::skyGradient(normalize(final vel)).
//   H = mean_i L_i (linear, per channel).
//   bloomStrength > 0: B = separable Gaussian blur of the whole HDR image H
//     (sigma = max(1, 0.01 * height) px, post::gaussianKernel1D, clamp-to-edge,
//     rows then columns, as post::blurRow), H' = post::bloomCombine(H, B, s).
//     Otherwise H' = H.
//   postProcess: srgb = post::postProcessPixel(H', exposure), packed with
//     post::packRgba8; else bytes = lroundf(255 * clamp01(H')), alpha 255.
// With spp 1 the offset is (0.5, 0.5), so the default config is bitwise the
// Phase-2 render.
//
// Tolerances: the CPU reference and the GPU differ by float rounding (fma
// contraction, powf), which is far below one byte for smooth pixels. Rays
// within a hair of the shadow edge or the photon sphere are chaotic, so a
// small fraction of channels may differ by more; hence "fraction within tol".

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "blackhole/camera.hpp"
#include "blackhole/geodesic.hpp"
#include "blackhole/image.hpp"
#include "blackhole/post.hpp"
#include "blackhole/render.hpp"
#include "cuda_test_support.hpp"

namespace {

using blackhole::Image;
using blackhole::RenderConfig;
namespace cam = blackhole::cam;
namespace geo = blackhole::geo;
namespace post = blackhole::post;

constexpr double kPi = 3.14159265358979323846;
constexpr double kBc = 2.598076211353316;  // 1.5 * sqrt(3)

RenderConfig makeConfig(int w, int h, float el, float exposure = 1.0f,
                        bool pp = false, int spp = 1, float bloom = 0.0f) {
  RenderConfig c;
  c.width = w;
  c.height = h;
  c.fovDegrees = 60.0f;
  c.elevationDegrees = el;
  c.observerDistance = 20.0f;
  c.postProcess = pp;
  c.exposure = exposure;
  c.spp = spp;
  c.bloomStrength = bloom;
  return c;
}

cam::Camera cameraOf(const RenderConfig& c) {
  return cam::makeCamera(c.width, c.height, c.observerDistance, c.fovDegrees,
                         c.elevationDegrees);
}

geo::Vec3 toGeo(cam::Vec3 v) { return {v.x, v.y, v.z}; }

// Linear radiance of one traced ray; `escaped` tells whether it was sky.
cam::Vec3 traceRadiance(const RenderConfig& config, const cam::Camera& camera,
                        float fx, float fy, bool* escaped = nullptr) {
  const geo::Ray ray = geo::makeRay(
      toGeo(camera.pos), toGeo(cam::primaryRayDirSub(camera, fx, fy)));
  const geo::TraceResult res =
      geo::trace(ray, blackhole::renderTraceParams(config));
  cam::Vec3 c{0.0f, 0.0f, 0.0f};
  const bool esc = res.outcome == geo::Outcome::Escaped;
  if (esc) {
    const geo::Vec3 d = geo::normalize(res.ray.vel);
    c = cam::skyGradient(cam::Vec3{d.x, d.y, d.z});
  }
  if (escaped) *escaped = esc;
  return c;
}

// HDR image (3 floats per pixel, row-major) of the supersampled trace.
std::vector<float> hdrReference(const RenderConfig& config) {
  const cam::Camera camera = cameraOf(config);
  std::vector<float> hdr(
      static_cast<std::size_t>(config.width) * config.height * 3, 0.0f);
  for (int y = 0; y < config.height; ++y) {
    for (int x = 0; x < config.width; ++x) {
      float sum[3] = {0, 0, 0};
      for (int i = 0; i < config.spp; ++i) {
        const post::SampleOffset o = post::sampleOffset(i, config.spp, x, y);
        const cam::Vec3 c = traceRadiance(config, camera, x + o.dx, y + o.dy);
        sum[0] += c.x;
        sum[1] += c.y;
        sum[2] += c.z;
      }
      const std::size_t k =
          (static_cast<std::size_t>(y) * config.width + x) * 3;
      for (int ch = 0; ch < 3; ++ch)
        hdr[k + ch] = sum[ch] / static_cast<float>(config.spp);
    }
  }
  return hdr;
}

// Separable clamp-to-edge Gaussian blur (rows, then columns) of every channel.
std::vector<float> blurImage(const std::vector<float>& img, int w, int h,
                             float sigma) {
  const std::vector<float> k = post::gaussianKernel1D(sigma);
  std::vector<float> tmp(img.size()), out(img.size());
  for (int ch = 0; ch < 3; ++ch) {
    for (int y = 0; y < h; ++y) {
      std::vector<float> row(static_cast<std::size_t>(w));
      for (int x = 0; x < w; ++x)
        row[x] = img[(static_cast<std::size_t>(y) * w + x) * 3 + ch];
      const std::vector<float> b = post::blurRow(row, k);
      for (int x = 0; x < w; ++x)
        tmp[(static_cast<std::size_t>(y) * w + x) * 3 + ch] = b[x];
    }
    for (int x = 0; x < w; ++x) {
      std::vector<float> col(static_cast<std::size_t>(h));
      for (int y = 0; y < h; ++y)
        col[y] = tmp[(static_cast<std::size_t>(y) * w + x) * 3 + ch];
      const std::vector<float> b = post::blurRow(col, k);
      for (int y = 0; y < h; ++y)
        out[(static_cast<std::size_t>(y) * w + x) * 3 + ch] = b[y];
    }
  }
  return out;
}

// H' = H (+ strength * blur(H)) for the config.
std::vector<float> combinedReference(const RenderConfig& config,
                                     const std::vector<float>& hdr) {
  if (!(config.bloomStrength > 0.0f)) return hdr;
  const float sigma =
      std::max(1.0f, 0.01f * static_cast<float>(config.height));
  const std::vector<float> blur =
      blurImage(hdr, config.width, config.height, sigma);
  std::vector<float> out(hdr.size());
  for (std::size_t i = 0; i < hdr.size(); i += 3) {
    const post::Vec3 r = post::bloomCombine(
        {hdr[i], hdr[i + 1], hdr[i + 2]}, {blur[i], blur[i + 1], blur[i + 2]},
        config.bloomStrength);
    out[i] = r.x;
    out[i + 1] = r.y;
    out[i + 2] = r.z;
  }
  return out;
}

// Final RGBA bytes of the reference pipeline.
std::vector<std::uint8_t> bytesReference(const RenderConfig& config,
                                         const std::vector<float>& comb) {
  std::vector<std::uint8_t> out(comb.size() / 3 * 4);
  for (std::size_t p = 0; p < comb.size() / 3; ++p) {
    const post::Vec3 c{comb[p * 3], comb[p * 3 + 1], comb[p * 3 + 2]};
    if (config.postProcess) {
      post::packRgba8(post::postProcessPixel(c, config.exposure), &out[p * 4]);
    } else {
      out[p * 4 + 0] = post::toByte(c.x);
      out[p * 4 + 1] = post::toByte(c.y);
      out[p * 4 + 2] = post::toByte(c.z);
      out[p * 4 + 3] = 255;
    }
  }
  return out;
}

std::vector<std::uint8_t> fullReference(const RenderConfig& config) {
  return bytesReference(config, combinedReference(config, hdrReference(config)));
}

// Fraction of RGB channels whose bytes differ by <= tol.
double fractionWithin(const Image& im, const std::vector<std::uint8_t>& ref,
                      int tol) {
  std::size_t ok = 0, total = 0;
  for (std::size_t p = 0; p < ref.size() / 4; ++p) {
    for (int ch = 0; ch < 3; ++ch) {
      const int d = std::abs(static_cast<int>(im.data()[p * 4 + ch]) -
                             static_cast<int>(ref[p * 4 + ch]));
      ok += d <= tol;
      ++total;
    }
  }
  return static_cast<double>(ok) / static_cast<double>(total);
}

int ch(const Image& im, int x, int y, int c) {
  return im.data()[im.pixelIndex(x, y) + c];
}

bool isBlack(const Image& im, int x, int y) {
  return ch(im, x, y, 0) == 0 && ch(im, x, y, 1) == 0 && ch(im, x, y, 2) == 0;
}

bool identical(const Image& a, const Image& b) {
  return a.width() == b.width() && a.height() == b.height() &&
         std::equal(a.data(), a.data() + a.size(), b.data());
}

// Shadow radius in pixels (see render_shadow_test.cpp for the derivation).
double shadowRadiusPx(const RenderConfig& c) {
  const double alphaC = std::asin(kBc / c.observerDistance);
  const double tanHalf = std::tan(c.fovDegrees * kPi / 180.0 * 0.5);
  return std::tan(alphaC) / tanHalf * (c.height / 2.0);
}

double centerDist(const RenderConfig& c, int x, int y) {
  return std::hypot(x + 0.5 - c.width / 2.0, y + 0.5 - c.height / 2.0);
}

// ---------------------------------------------------------------------------
// 1. Regression: the default config is the Phase-2 render, bitwise.
// ---------------------------------------------------------------------------

TEST(RenderPostTest, DefaultConfigIsUnchangedPhase2Render) {
  SKIP_IF_NO_CUDA_DEVICE();
  RenderConfig def;
  def.width = 64;
  def.height = 48;
  def.elevationDegrees = 30.0f;
  RenderConfig explicitCfg = def;
  explicitCfg.postProcess = false;
  explicitCfg.spp = 1;
  explicitCfg.bloomStrength = 0.0f;
  explicitCfg.exposure = 1.0f;

  const Image a = blackhole::render(def);
  const Image b = blackhole::render(explicitCfg);
  EXPECT_TRUE(identical(a, b));

  // Phase-2 reference: bytes = lroundf(255 * c), centre rays, alpha 255.
  const cam::Camera camera = cameraOf(def);
  std::size_t ok = 0, total = 0;
  for (int y = 0; y < def.height; ++y) {
    for (int x = 0; x < def.width; ++x) {
      const cam::Vec3 c = traceRadiance(def, camera, x + 0.5f, y + 0.5f);
      const float v[3] = {c.x, c.y, c.z};
      for (int k = 0; k < 3; ++k) {
        ok += std::abs(ch(a, x, y, k) -
                       static_cast<int>(std::lround(255.0f * v[k]))) <= 1;
        ++total;
      }
      EXPECT_EQ(ch(a, x, y, 3), 255);
    }
  }
  // +-1: fma/rounding differences on a .5 boundary; chaotic edge rays rare.
  EXPECT_GE(static_cast<double>(ok) / total, 0.995);
}

// ---------------------------------------------------------------------------
// 2. Tone mapping only.
// ---------------------------------------------------------------------------

TEST(RenderPostTest, ToneMappingMatchesReferenceAndKeepsShadowBlack) {
  SKIP_IF_NO_CUDA_DEVICE();
  for (const float el : {0.0f, 30.0f}) {
    for (const float exposure : {1.0f, 0.5f, 2.0f, 0.05f}) {  // 0.05: dark end, sRGB linear toe vs gamma
      const RenderConfig config = makeConfig(64, 48, el, exposure, true);
      const Image im = blackhole::render(config);
      ASSERT_EQ(im.width(), 64);
      ASSERT_EQ(im.height(), 48);
      const std::vector<std::uint8_t> ref = fullReference(config);
      // ACES + sRGB are smooth (slope <= ~2.5 byte/byte near black), float
      // powf differences are < 1 byte; +-2 absorbs them.
      EXPECT_GE(fractionWithin(im, ref, 2), 0.99)
          << "el " << el << " exposure " << exposure;
      int black = 0;
      for (int y = 0; y < 48; ++y) {
        for (int x = 0; x < 64; ++x) {
          EXPECT_EQ(ch(im, x, y, 3), 255);
          // Reference-black pixels (shadow interior) must be exactly black.
          const std::size_t p = (static_cast<std::size_t>(y) * 64 + x) * 4;
          if (ref[p] == 0 && ref[p + 1] == 0 && ref[p + 2] == 0 &&
              centerDist(config, x, y) < shadowRadiusPx(config) - 1.0) {
            ++black;
            EXPECT_TRUE(isBlack(im, x, y)) << x << "," << y;
          }
        }
      }
      EXPECT_GT(black, 10) << "non-vacuity: shadow interior is populated";
      EXPECT_TRUE(isBlack(im, 32, 24));
    }
  }
}

TEST(RenderPostTest, ExposureIsMonotoneAndPostProcessChangesImage) {
  SKIP_IF_NO_CUDA_DEVICE();
  const Image dim = blackhole::render(makeConfig(64, 48, 0.0f, 0.5f, true));
  const Image mid = blackhole::render(makeConfig(64, 48, 0.0f, 1.0f, true));
  const Image bright = blackhole::render(makeConfig(64, 48, 0.0f, 2.0f, true));
  bool strictDim = false, strictBright = false;
  for (std::size_t i = 0; i < mid.size(); ++i) {
    if (i % 4 == 3) continue;
    // Same traced HDR value, ACES and sRGB are monotone: exact >=.
    EXPECT_LE(dim.data()[i], mid.data()[i]) << i;
    EXPECT_LE(mid.data()[i], bright.data()[i]) << i;
    strictDim = strictDim || dim.data()[i] < mid.data()[i];
    strictBright = strictBright || mid.data()[i] < bright.data()[i];
  }
  EXPECT_TRUE(strictDim);
  EXPECT_TRUE(strictBright);

  const Image raw = blackhole::render(makeConfig(64, 48, 0.0f, 1.0f, false));
  EXPECT_FALSE(identical(raw, mid));
  // Tone mapping visibly changes mid-grey: count channels differing by > 8.
  std::size_t diff = 0;
  for (std::size_t i = 0; i < raw.size(); ++i)
    if (i % 4 != 3 && std::abs(raw.data()[i] - mid.data()[i]) > 8) ++diff;
  EXPECT_GT(diff, raw.size() / 4);
}

// ---------------------------------------------------------------------------
// 3. Supersampling (postProcess = false, bloom = 0).
// ---------------------------------------------------------------------------

TEST(RenderPostTest, SupersamplingMatchesReference) {
  SKIP_IF_NO_CUDA_DEVICE();
  for (const int spp : {1, 4, 9, 16}) {
    const RenderConfig config = makeConfig(48, 36, 0.0f, 1.0f, false, spp);
    const Image im = blackhole::render(config);
    const std::vector<std::uint8_t> ref = fullReference(config);
    // Edge pixels average chaotic samples near the shadow; the vast majority
    // of channels are smooth and agree to well under 2 bytes.
    EXPECT_GE(fractionWithin(im, ref, 2), 0.99) << "spp " << spp;
    for (int y = 0; y < 36; ++y)
      for (int x = 0; x < 48; ++x) ASSERT_EQ(ch(im, x, y, 3), 255);
  }
}

// Antialiasing. With spp 1 every pixel is either black or the sky colour of
// its single ray, so no non-black pixel is darker than 0.8x its own sky
// colour. With spp 16 pixels straddling the shadow edge contain a mix of
// captured (black) and escaped samples and are darker. "Intermediate" =
// non-black pixel whose RGB sum is < 0.8 x the sum of the centre ray's sky
// colour, or non-black pixel whose centre ray is captured.
int countIntermediate(const RenderConfig& config, const Image& im) {
  const cam::Camera camera = cameraOf(config);
  int n = 0;
  for (int y = 0; y < config.height; ++y) {
    for (int x = 0; x < config.width; ++x) {
      bool esc = false;
      const cam::Vec3 c = traceRadiance(config, camera, x + 0.5f, y + 0.5f, &esc);
      if (isBlack(im, x, y)) continue;
      const float sum = 255.0f * (c.x + c.y + c.z);
      const int got = ch(im, x, y, 0) + ch(im, x, y, 1) + ch(im, x, y, 2);
      if (!esc || got < 0.8f * sum) ++n;
    }
  }
  return n;
}

TEST(RenderPostTest, SupersamplingAntialiasesShadowEdge) {
  SKIP_IF_NO_CUDA_DEVICE();
  const RenderConfig c1 = makeConfig(96, 72, 0.0f, 1.0f, false, 1);
  const RenderConfig c16 = makeConfig(96, 72, 0.0f, 1.0f, false, 16);
  const Image i1 = blackhole::render(c1);
  const Image i16 = blackhole::render(c16);

  // Shadow radius ~8.2 px -> ~50 boundary pixels, a good fraction partially
  // covered (>= 1/16 black coverage), so many intermediate pixels at spp 16.
  const int n1 = countIntermediate(c1, i1);
  const int n16 = countIntermediate(c16, i16);
  EXPECT_LE(n1, 2);  // binary classification: ~0
  EXPECT_GE(n16, 10);
  EXPECT_GE(n16, 3 * std::max(n1, 1));

  // Mean luminance is nearly unchanged (area of the shadow differs by < 1%
  // and the smooth gradient averages to itself).
  double s1 = 0, s16 = 0;
  for (std::size_t i = 0; i < i1.size(); ++i) {
    if (i % 4 == 3) continue;
    s1 += i1.data()[i];
    s16 += i16.data()[i];
  }
  EXPECT_NEAR(s16, s1, 0.02 * s1);

  // Far from the shadow the sky gradient is smooth: one pixel changes the
  // colour by < 1.5 byte (0.5 * dd * 255 with dd <= 0.7 * fov/h ~ 0.01 plus
  // modest lensing), so averaging jittered samples stays within 3 bytes.
  const double rpx = shadowRadiusPx(c1);
  int far = 0, farBad = 0, nearDiff = 0;
  for (int y = 0; y < 72; ++y) {
    for (int x = 0; x < 96; ++x) {
      int maxd = 0;
      for (int k = 0; k < 3; ++k)
        maxd = std::max(maxd, std::abs(ch(i1, x, y, k) - ch(i16, x, y, k)));
      const double r = centerDist(c1, x, y);
      if (r > 2.5 * rpx) {
        ++far;
        farBad += maxd > 3;
      } else if (r < rpx + 2.0) {
        nearDiff += maxd > 3;
      }
    }
  }
  EXPECT_GT(far, 1000);
  EXPECT_EQ(farBad, 0);
  EXPECT_GE(nearDiff, 5) << "spp must change the edge pixels";
}

TEST(RenderPostTest, SupersamplingIsDeterministicAndSppDependent) {
  SKIP_IF_NO_CUDA_DEVICE();
  const RenderConfig config = makeConfig(48, 36, 30.0f, 1.0f, false, 4);
  EXPECT_TRUE(identical(blackhole::render(config), blackhole::render(config)));
  RenderConfig c9 = config;
  c9.spp = 9;
  EXPECT_FALSE(identical(blackhole::render(config), blackhole::render(c9)));
}

// ---------------------------------------------------------------------------
// 4. Bloom (postProcess = false, spp = 1).
// ---------------------------------------------------------------------------

TEST(RenderPostTest, BloomMatchesReferenceAndZeroStrengthIsNoBloom) {
  SKIP_IF_NO_CUDA_DEVICE();
  for (const float strength : {0.2f, 1.0f}) {
    const RenderConfig config =
        makeConfig(40, 30, 0.0f, 1.0f, false, 1, strength);
    const Image im = blackhole::render(config);
    EXPECT_GE(fractionWithin(im, fullReference(config), 2), 0.99)
        << "strength " << strength;
  }
  // Tall image: sigma = 0.01 * 192 = 1.92 px (> the 1 px floor), kernel radius
  // 6. A fixed sigma of 1 or a wrong scale changes the edge glow by > 2 bytes.
  const RenderConfig tall = makeConfig(256, 192, 0.0f, 1.0f, false, 1, 1.0f);
  EXPECT_GE(fractionWithin(blackhole::render(tall), fullReference(tall), 2),
            0.99);
  RenderConfig zero = makeConfig(40, 30, 0.0f, 1.0f, false, 1, 0.0f);
  EXPECT_TRUE(identical(blackhole::render(zero),
                        blackhole::render(makeConfig(40, 30, 0.0f))));
}

TEST(RenderPostTest, BloomOnlyBrightensAndGlowsNextToTheShadow) {
  SKIP_IF_NO_CUDA_DEVICE();
  const RenderConfig base = makeConfig(96, 72, 0.0f);
  RenderConfig bloomed = base;
  bloomed.bloomStrength = 1.0f;
  const Image a = blackhole::render(base);
  const Image b = blackhole::render(bloomed);

  // H' = H + s*B with B >= 0 and the same H: bytes can only grow (rounding
  // and clamping are monotone).
  bool strict = false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    EXPECT_LE(a.data()[i], b.data()[i]) << i;
    strict = strict || a.data()[i] < b.data()[i];
  }
  EXPECT_TRUE(strict);

  // Sigma = 1 px (kernel radius 3): shadow pixels adjacent to the sky pick up
  // light; the deep interior (> 4 px from the edge) has B == 0 exactly.
  const double rpx = shadowRadiusPx(base);
  ASSERT_GT(rpx, 8.0);
  int lit = 0, deepLit = 0, deep = 0;
  for (int y = 0; y < 72; ++y) {
    for (int x = 0; x < 96; ++x) {
      const double r = centerDist(base, x, y);
      if (isBlack(a, x, y) && !isBlack(b, x, y)) {
        ++lit;
        EXPECT_GT(r, rpx - 4.0) << "glow must stay near the edge";
      }
      if (r < rpx - 4.0) {
        ++deep;
        deepLit += !isBlack(b, x, y);
      }
    }
  }
  EXPECT_GT(deep, 10);
  EXPECT_EQ(deepLit, 0);
  EXPECT_GE(lit, 20);
  EXPECT_TRUE(isBlack(b, 48, 36));
}

// Energy: the reference's total H' equals (1+s)*sum(H) up to the clamp-to-edge
// border duplication (a blur with clamped borders conserves mass except where
// the border differs from its replicated neighbours: < 3% here). The GPU bytes
// must reproduce the clamped reference total (rounding is unbiased: < 1%).
TEST(RenderPostTest, BloomEnergyIsConserved) {
  SKIP_IF_NO_CUDA_DEVICE();
  const float s = 0.2f;
  const RenderConfig config = makeConfig(40, 30, 0.0f, 1.0f, false, 1, s);
  const std::vector<float> hdr = hdrReference(config);
  const std::vector<float> comb = combinedReference(config, hdr);
  double sumH = 0, sumC = 0, sumClamped = 0;
  for (std::size_t i = 0; i < hdr.size(); ++i) {
    sumH += hdr[i];
    sumC += comb[i];
    sumClamped += std::min(1.0f, comb[i]);
  }
  EXPECT_NEAR(sumC, (1.0 + s) * sumH, 0.03 * (1.0 + s) * sumH);
  EXPECT_LT(sumClamped, sumC);  // the gradient sky does saturate: clamp cuts

  const Image im = blackhole::render(config);
  double sumBytes = 0;
  for (std::size_t i = 0; i < im.size(); ++i)
    if (i % 4 != 3) sumBytes += im.data()[i];
  EXPECT_NEAR(sumBytes / 255.0, sumClamped, 0.01 * sumClamped);
  // Bloom adds energy over the plain image.
  EXPECT_GT(sumBytes / 255.0, sumH);
}

// Image sizes that are not multiples of the 16x16 launch block, so GPU blur
// border/index bugs show up. Sigma is 1 px here, so the blur is weak; the
// tone-mapped variant amplifies differences near black. Note: clamp-to-edge
// and mirror padding are indistinguishable at this tolerance (the border
// colour gradient is smooth), which is accepted.
TEST(RenderPostTest, BloomMatchesReferenceOnOddSizes) {
  SKIP_IF_NO_CUDA_DEVICE();
  for (const auto wh : {std::pair<int, int>{37, 29}, std::pair<int, int>{50, 21}}) {
    for (const bool pp : {false, true}) {
      const RenderConfig config =
          makeConfig(wh.first, wh.second, 30.0f, 1.0f, pp, 1, 1.0f);
      const Image im = blackhole::render(config);
      ASSERT_EQ(im.width(), wh.first);
      EXPECT_GE(fractionWithin(im, fullReference(config), 2), 0.99)
          << wh.first << "x" << wh.second << " pp " << pp;
    }
  }
}

TEST(RenderPostTest, SupersamplingMatchesReferenceForLargerSpp) {
  SKIP_IF_NO_CUDA_DEVICE();
  for (const int spp : {25, 36, 49}) {
    const RenderConfig config = makeConfig(24, 18, 0.0f, 1.0f, false, spp);
    EXPECT_GE(fractionWithin(blackhole::render(config), fullReference(config), 2),
              0.99)
        << "spp " << spp;
  }
}

// ---------------------------------------------------------------------------
// 5. Combined pipeline.
// ---------------------------------------------------------------------------

TEST(RenderPostTest, CombinedPipelineMatchesReference) {
  SKIP_IF_NO_CUDA_DEVICE();
  const RenderConfig config = makeConfig(32, 24, 30.0f, 1.5f, true, 4, 0.2f);
  const Image im = blackhole::render(config);
  // +-3 / 98%: jittered edge samples + blur spread chaotic edge differences.
  EXPECT_GE(fractionWithin(im, fullReference(config), 3), 0.98);
  for (int y = 0; y < 24; ++y)
    for (int x = 0; x < 32; ++x) ASSERT_EQ(ch(im, x, y, 3), 255);

  // Strong bloom pushes H' well above 1 near the edge; the HDR value must
  // reach the tone mapper unclamped (ACES(2) = 0.97 vs ACES(1) = 0.80).
  const RenderConfig hot = makeConfig(32, 24, 0.0f, 1.0f, true, 1, 1.0f);
  EXPECT_GE(fractionWithin(blackhole::render(hot), fullReference(hot), 3),
            0.98);
}

// ---------------------------------------------------------------------------
// 6. Edge cases.
// ---------------------------------------------------------------------------

TEST(RenderPostTest, MaxSamplesPerPixelIsFast) {
  SKIP_IF_NO_CUDA_DEVICE();
  blackhole::render(makeConfig(8, 8, 0.0f));  // warm up the CUDA context
  const RenderConfig config = makeConfig(64, 48, 0.0f, 1.0f, true, 64, 0.2f);
  const auto t0 = std::chrono::steady_clock::now();
  const Image im = blackhole::render(config);
  const double secs = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - t0).count();
  EXPECT_LT(secs, 2.0);
  EXPECT_EQ(im.width(), 64);
  EXPECT_TRUE(isBlack(im, 32, 24));
  EXPECT_FALSE(isBlack(im, 2, 2));

  // Not a vacuous pass: spp 64 agrees with the reference (small image keeps
  // the 64-sample CPU trace cheap).
  const RenderConfig small = makeConfig(16, 12, 0.0f, 1.0f, true, 64, 0.2f);
  EXPECT_GE(fractionWithin(blackhole::render(small), fullReference(small), 3),
            0.97);
}

TEST(RenderPostTest, InvalidSppThrows) {
  for (const int spp : {0, 2, 3, 65, -4}) {
    RenderConfig c = makeConfig(8, 8, 0.0f);
    c.spp = spp;
    EXPECT_THROW(blackhole::render(c), std::invalid_argument) << spp;
  }
}

TEST(RenderPostTest, HugeExposureSaturatesWithoutNaN) {
  SKIP_IF_NO_CUDA_DEVICE();
  const RenderConfig hot = makeConfig(48, 36, 30.0f, 1.0e4f, true);
  const Image im = blackhole::render(hot);
  const Image raw = blackhole::render(makeConfig(48, 36, 30.0f));
  int saturated = 0, black = 0;
  for (int y = 0; y < 36; ++y) {
    for (int x = 0; x < 48; ++x) {
      EXPECT_EQ(ch(im, x, y, 3), 255);
      if (isBlack(raw, x, y)) {
        ++black;
        EXPECT_TRUE(isBlack(im, x, y)) << x << "," << y;
        continue;
      }
      for (int k = 0; k < 3; ++k) {
        // A raw byte >= 2 means linear >= 0.0078, times 1e4 >= 78: ACES is
        // fully saturated (>= 1), so sRGB is 255. NaN would not give 255.
        if (ch(raw, x, y, k) >= 2) {
          EXPECT_EQ(ch(im, x, y, k), 255) << x << "," << y << "," << k;
          ++saturated;
        }
      }
    }
  }
  EXPECT_GT(black, 0);
  EXPECT_GT(saturated, 1000);
}

}  // namespace
