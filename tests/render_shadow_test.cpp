// Phase-2 tests of the lensing render(): every pixel traces the primary ray
// through the Schwarzschild geodesic integrator (geodesic.hpp) with
// renderTraceParams(config); Horizon and MaxSteps -> (0,0,0), Escaped ->
// skyGradient(normalize(final velocity)); bytes = round(255*c), alpha 255.
//
// Shadow geometry used throughout. The camera sits at distance D from the
// origin and looks at it, so a ray at angle alpha from the forward axis has
// impact parameter b = D*sin(alpha) (exactly, for the straight line through the
// camera position; the integrator conserves h2 = |pos x vel|^2 = b^2). Rays
// start moving inward at r = D > 1.5 rs, so capture happens iff
// b < b_c = 1.5*sqrt(3) rs. With pixel centre offsets (dx, dy) from the image
// centre (in pixels) the ray satisfies
//   tan(alpha) = hypot(dx, dy) / (H/2) * tanHalfFovY
// (because ndcX*aspect*t = dx*2/H*t and ndcY*t = dy*2/H*t), hence the shadow is
// a disc of radius r_px = tan(alpha_c) / tanHalfFovY * (H/2) pixels, with
// alpha_c = asin(b_c / D).
//
// "Black" means all of R, G, B == 0. A sky pixel is never black: the colour
// channel is 0.5 + 0.5*d_i, which rounds to byte 0 only if d_i < -0.996, and a
// unit vector can have at most one component that negative, so at least two
// channels are >= 127. Therefore black == Horizon or MaxSteps.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "blackhole/camera.hpp"
#include "blackhole/geodesic.hpp"
#include "blackhole/image.hpp"
#include "blackhole/render.hpp"
#include "cuda_test_support.hpp"

namespace {

using blackhole::Image;
using blackhole::RenderConfig;
namespace cam = blackhole::cam;
namespace geo = blackhole::geo;

constexpr double kPi = 3.14159265358979323846;
constexpr double kBc = 2.598076211353316;  // 1.5 * sqrt(3)

RenderConfig makeConfig(int w, int h, float fov, float el, float dist) {
  RenderConfig c;
  c.width = w;
  c.height = h;
  c.fovDegrees = fov;
  c.elevationDegrees = el;
  c.observerDistance = dist;
  return c;
}

std::uint8_t quantize(float c) {
  return static_cast<std::uint8_t>(std::lround(255.0f * c));
}

bool isBlack(const Image& im, int x, int y) {
  const std::size_t i = im.pixelIndex(x, y);
  return im.data()[i] == 0 && im.data()[i + 1] == 0 && im.data()[i + 2] == 0;
}

// Analytic shadow radius in pixels (see file comment), in double precision.
double analyticShadowRadiusPx(const RenderConfig& c) {
  const double alphaC = std::asin(kBc / c.observerDistance);
  const double tanHalf = std::tan(c.fovDegrees * kPi / 180.0 * 0.5);
  return std::tan(alphaC) / tanHalf * (c.height / 2.0);
}

double centerDist(const RenderConfig& c, int x, int y) {
  return std::hypot(x + 0.5 - c.width / 2.0, y + 0.5 - c.height / 2.0);
}

std::size_t blackCount(const Image& im) {
  std::size_t n = 0;
  for (int y = 0; y < im.height(); ++y)
    for (int x = 0; x < im.width(); ++x) n += isBlack(im, x, y);
  return n;
}

geo::Vec3 toGeo(cam::Vec3 v) { return {v.x, v.y, v.z}; }

// Everything the CPU reference needs for one pixel.
struct CpuPixel {
  geo::TraceResult result;
  cam::Vec3 primary;
  std::uint8_t rgb[3];
};

CpuPixel cpuTrace(const RenderConfig& config, const cam::Camera& camera, int x,
                  int y) {
  CpuPixel p;
  p.primary = cam::primaryRayDir(camera, x, y);
  const geo::Ray ray = geo::makeRay(toGeo(camera.pos), toGeo(p.primary));
  p.result = geo::trace(ray, blackhole::renderTraceParams(config));
  p.rgb[0] = p.rgb[1] = p.rgb[2] = 0;
  if (p.result.outcome == geo::Outcome::Escaped) {
    const geo::Vec3 d = geo::normalize(p.result.ray.vel);
    const cam::Vec3 c = cam::skyGradient(cam::Vec3{d.x, d.y, d.z});
    p.rgb[0] = quantize(c.x);
    p.rgb[1] = quantize(c.y);
    p.rgb[2] = quantize(c.z);
  }
  return p;
}

cam::Camera cameraOf(const RenderConfig& c) {
  return cam::makeCamera(c.width, c.height, c.observerDistance, c.fovDegrees,
                         c.elevationDegrees);
}

int channel(const Image& im, int x, int y, int ch) {
  return im.data()[im.pixelIndex(x, y) + ch];
}

// ---------------------------------------------------------------------------
// renderTraceParams (CPU only)
// ---------------------------------------------------------------------------

TEST(RenderTraceParamsTest, EscapeRadiusIsMaxOf40AndTwiceDistance) {
  struct Case {
    float dist, rEscape;
  };
  for (const Case c : {Case{20.0f, 40.0f}, Case{100.0f, 200.0f},
                       Case{10.0f, 40.0f}, Case{40.0f, 80.0f},
                       Case{25.0f, 50.0f}, Case{5.0f, 40.0f},
                       Case{20.5f, 41.0f}}) {
    const geo::TraceParams p =
        blackhole::renderTraceParams(makeConfig(8, 8, 60.0f, 0.0f, c.dist));
    EXPECT_FLOAT_EQ(p.rEscape, c.rEscape) << "distance " << c.dist;
    EXPECT_EQ(p.maxSteps, 4000) << "distance " << c.dist;
    EXPECT_FALSE(p.nearDisk) << "distance " << c.dist;
  }
}

// Stall guard. Only rays whose b is within a hair above b_c orbit the photon
// sphere long enough to exhaust 4000 steps (each step is <= 0.7 and, near
// r = 1.5, ~0.18; escape needs only a few photon-sphere turns unless
// ln(1/(b/b_c - 1)) is huge). Such rays occupy an annulus much thinner than a
// pixel; we allow < 0.5% of the pixels (about 80 of 16384) to be MaxSteps.
// Escaped and Horizon must both be well populated (non-vacuity).
TEST(RenderTraceParamsTest, AlmostNoRayExhaustsStepBudget) {
  for (const float el : {0.0f, 30.0f}) {
    const RenderConfig config = makeConfig(128, 128, 60.0f, el, 20.0f);
    const cam::Camera camera = cameraOf(config);
    int horizon = 0, escaped = 0, maxSteps = 0;
    for (int y = 0; y < 128; ++y) {
      for (int x = 0; x < 128; ++x) {
        switch (cpuTrace(config, camera, x, y).result.outcome) {
          case geo::Outcome::Horizon: ++horizon; break;
          case geo::Outcome::Escaped: ++escaped; break;
          case geo::Outcome::MaxSteps: ++maxSteps; break;
        }
      }
    }
    EXPECT_LT(maxSteps, 128 * 128 * 5 / 1000) << "elevation " << el;
    // Analytic shadow radius at this size: 14.5 px -> area ~ 660 px.
    EXPECT_GT(horizon, 550) << "elevation " << el;
    EXPECT_GT(escaped, 128 * 128 / 2) << "elevation " << el;
  }
}

// ---------------------------------------------------------------------------
// Lensing direction (CPU only, geo::trace)
// ---------------------------------------------------------------------------

// Light bends towards the hole. Camera at el 0 on the -z axis looking +z with
// right = +x. A ray on the +x side (outside the shadow) is deflected towards -x
// (towards the hole, which is at x = 0), so the escaped direction's x is
// smaller than the unlensed one; on the -x side it is larger. The deflection
// is >= ~2 rs/b rad (b <= 6 here, so >= 0.33 rad) and dir.x changes by far
// more than float noise, so strict inequality with a 0.05 margin is safe.
TEST(RenderLensingTest, RaysBendTowardsTheHole) {
  const RenderConfig config = makeConfig(400, 400, 60.0f, 0.0f, 20.0f);
  const cam::Camera camera = cameraOf(config);
  const double rpx = analyticShadowRadiusPx(config);
  for (const double mult : {1.1, 1.5, 2.0}) {
    const int off = static_cast<int>(std::lround(mult * rpx));
    for (const int sign : {+1, -1}) {
      const int x = 200 + (sign > 0 ? off : -off - 1);  // symmetric centres
      const CpuPixel p = cpuTrace(config, camera, x, 200);
      ASSERT_EQ(p.result.outcome, geo::Outcome::Escaped)
          << "mult " << mult << " sign " << sign;
      const geo::Vec3 d = geo::normalize(p.result.ray.vel);
      if (sign > 0) {
        EXPECT_LT(d.x, p.primary.x - 0.05f) << "mult " << mult;
      } else {
        EXPECT_GT(d.x, p.primary.x + 0.05f) << "mult " << mult;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// GPU tests
// ---------------------------------------------------------------------------

struct ShadowCase {
  int w, h;
  float fov, el, dist;
};

class RenderShadowGeometryTest : public ::testing::TestWithParam<ShadowCase> {};

// Pixels whose centre is > 2 px inside r_px must be black, > 2 px outside must
// not be. The 2 px margin covers float error in the ray angle and the
// quantisation of a hard edge; b changes by ~0.06..0.12 per pixel here, so the
// margin is >= 4% in b, far above integrator error, and also far from the
// slow-orbit (MaxSteps) annulus. The black area must equal pi*r_px^2 within 3%
// (the discretisation error of counting pixel centres in a disc of radius r is
// O(r^0.65) pixels, <= ~1% of the area for r >= 20 px).
TEST_P(RenderShadowGeometryTest, ShadowIsADiscOfAnalyticRadius) {
  SKIP_IF_NO_CUDA_DEVICE();
  const ShadowCase p = GetParam();
  const RenderConfig config = makeConfig(p.w, p.h, p.fov, p.el, p.dist);
  const double rpx = analyticShadowRadiusPx(config);
  ASSERT_GT(rpx, 20.0);  // test design: shadow big enough to measure

  const Image image = blackhole::render(config);
  ASSERT_EQ(image.width(), p.w);
  ASSERT_EQ(image.height(), p.h);

  int insideWrong = 0, outsideWrong = 0, badAlpha = 0;
  for (int y = 0; y < p.h; ++y) {
    for (int x = 0; x < p.w; ++x) {
      const double r = centerDist(config, x, y);
      if (channel(image, x, y, 3) != 255) ++badAlpha;
      if (r < rpx - 2.0 && !isBlack(image, x, y)) ++insideWrong;
      if (r > rpx + 2.0 && isBlack(image, x, y)) ++outsideWrong;
    }
  }
  EXPECT_EQ(badAlpha, 0);
  EXPECT_EQ(insideWrong, 0) << "r_px = " << rpx;
  EXPECT_EQ(outsideWrong, 0) << "r_px = " << rpx;

  const double area = static_cast<double>(blackCount(image));
  const double expected = kPi * rpx * rpx;
  EXPECT_NEAR(area, expected, 0.03 * expected) << "r_px = " << rpx;

  // The centre ray passes through the origin.
  EXPECT_TRUE(isBlack(image, p.w / 2, p.h / 2));
}

INSTANTIATE_TEST_SUITE_P(
    Shadow, RenderShadowGeometryTest,
    ::testing::Values(ShadowCase{400, 400, 60.0f, 0.0f, 20.0f},
                      ShadowCase{400, 400, 60.0f, 30.0f, 20.0f},
                      ShadowCase{400, 400, 90.0f, 0.0f, 10.0f},
                      ShadowCase{400, 400, 60.0f, 30.0f, 40.0f},
                      ShadowCase{400, 400, 90.0f, 30.0f, 20.0f},
                      ShadowCase{480, 270, 60.0f, 0.0f, 20.0f},
                      ShadowCase{480, 270, 90.0f, 30.0f, 10.0f}));

// At el = 0 the scene is rotationally symmetric about the view axis, so the
// black mask of a square image is invariant under a 90 degree rotation about
// the image centre: pixel (x, y) -> (W-1-y, x) maps centres to centres exactly.
// Float noise can flip only pixels whose centre is within ~1e-5 of the edge; we
// allow at most 2% of the boundary pixels (black pixels with a non-black
// 4-neighbour) to differ -- in practice 0.
TEST(RenderShadowTest, MaskIsInvariantUnderQuarterTurn) {
  SKIP_IF_NO_CUDA_DEVICE();
  for (const float dist : {20.0f, 10.0f}) {
    const int n = 400;
    const Image im = blackhole::render(makeConfig(n, n, 60.0f, 0.0f, dist));
    int boundary = 0, mismatches = 0;
    for (int y = 0; y < n; ++y) {
      for (int x = 0; x < n; ++x) {
        const bool b = isBlack(im, x, y);
        if (b != isBlack(im, n - 1 - y, x)) ++mismatches;
        if (b) {
          const bool edge = (x > 0 && !isBlack(im, x - 1, y)) ||
                            (x < n - 1 && !isBlack(im, x + 1, y)) ||
                            (y > 0 && !isBlack(im, x, y - 1)) ||
                            (y < n - 1 && !isBlack(im, x, y + 1));
          boundary += edge;
        }
      }
    }
    ASSERT_GT(boundary, 100) << "distance " << dist;  // shadow exists
    EXPECT_LE(mismatches, boundary * 2 / 100) << "distance " << dist;
  }
}

// Closer observer -> larger angular (hence pixel) shadow. Radius is measured
// from the black area, r = sqrt(area/pi), and must also match the analytic one
// within 3% (area within 6%).
TEST(RenderShadowTest, ShadowShrinksAsObserverMovesAway) {
  SKIP_IF_NO_CUDA_DEVICE();
  double prev = 1e9;
  for (const float dist : {10.0f, 20.0f, 40.0f}) {
    const RenderConfig config = makeConfig(400, 400, 60.0f, 0.0f, dist);
    const Image im = blackhole::render(config);
    const double r = std::sqrt(static_cast<double>(blackCount(im)) / kPi);
    EXPECT_LT(r, prev) << "distance " << dist;
    EXPECT_NEAR(r, analyticShadowRadiusPx(config),
                0.03 * analyticShadowRadiusPx(config))
        << "distance " << dist;
    prev = r;
  }
}

class RenderCpuGpuAgreementTest : public ::testing::TestWithParam<ShadowCase> {
};

// The CPU reference uses the very same camera.hpp / geodesic.hpp calls and
// renderTraceParams. Host vs device builds differ in FMA contraction and
// sinf/tanf/sqrt implementations (a few ulp); integrated over ~100 steps this
// can move a ray's final direction by ~1e-5 (<< 1 byte step of 0.002), except
// for rays within ~1e-4 of b_c, whose outcome (horizon vs escape / orbit length)
// can flip. So we require >= 99% of channel values within +-2 bytes and at most
// 0.5% of pixels whose black/non-black classification differs.
TEST_P(RenderCpuGpuAgreementTest, MatchesCpuReference) {
  SKIP_IF_NO_CUDA_DEVICE();
  const ShadowCase p = GetParam();
  const RenderConfig config = makeConfig(p.w, p.h, p.fov, p.el, p.dist);
  const cam::Camera camera = cameraOf(config);
  const Image image = blackhole::render(config);
  ASSERT_EQ(image.width(), p.w);
  ASSERT_EQ(image.height(), p.h);

  int channelsOff = 0, classDiff = 0;
  for (int y = 0; y < p.h; ++y) {
    for (int x = 0; x < p.w; ++x) {
      const CpuPixel cpu = cpuTrace(config, camera, x, y);
      ASSERT_EQ(channel(image, x, y, 3), 255) << x << "," << y;
      bool cpuBlack = true, gpuBlack = true;
      for (int ch = 0; ch < 3; ++ch) {
        const int g = channel(image, x, y, ch);
        if (std::abs(g - int{cpu.rgb[ch]}) > 2) ++channelsOff;
        cpuBlack = cpuBlack && cpu.rgb[ch] == 0;
        gpuBlack = gpuBlack && g == 0;
      }
      classDiff += cpuBlack != gpuBlack;
    }
  }
  const int total = p.w * p.h;
  EXPECT_LE(channelsOff, total * 3 / 100);
  EXPECT_LE(classDiff, total * 5 / 1000);
}

INSTANTIATE_TEST_SUITE_P(
    CpuGpu, RenderCpuGpuAgreementTest,
    ::testing::Values(ShadowCase{64, 48, 60.0f, 0.0f, 20.0f},
                      ShadowCase{64, 48, 60.0f, 30.0f, 20.0f},
                      ShadowCase{64, 48, 90.0f, 30.0f, 10.0f},
                      ShadowCase{64, 48, 60.0f, 10.0f, 40.0f},
                      // Huge distances: with the spec params (maxSteps 4000,
                      // rEscape 2D) a ray from D=1500 needs > 4300 steps, so
                      // every pixel is MaxSteps -> black; D=1000 mixes
                      // MaxSteps and Escaped pixels. A kernel that ignores
                      // renderTraceParams (default 2000 steps / rEscape 40) or
                      // colours MaxSteps as sky flips the black/non-black
                      // classification of most pixels.
                      ShadowCase{32, 24, 60.0f, 0.0f, 1500.0f},
                      ShadowCase{32, 24, 60.0f, 30.0f, 1000.0f}));

// Lensing is real, near the shadow and far from it.
//  * Just outside the edge (1.1 * r_px along +x; b ~ 1.1 b_c = 2.9 rs) the ray
//    is deflected by ~1 rad or more, so the colour differs from the flat image
//    by far more than 30/255 in some channel.
//  * Far corner of a D = 40, fov 60 image: alpha = atan(sqrt2 * tan 30) = 39.2
//    deg, b = 40 sin(alpha) = 25.3 rs, weak-field deflection ~ 2 rs/b = 0.079
//    rad at infinity; the rays here start at r = 40 and are read at r = 80, so
//    only part of it accrues. The colour 0.5+0.5*d moves by at most
//    0.5 * 0.079 * 255 = 10.1 bytes per channel (|delta d| <= deflection), and
//    a reference implementation measures 7 (the blue channel), so we require
//    <= 10 bytes: the analytic upper bound, not a fitted number.
TEST(RenderLensingTest, StrongNearShadowAndWeakInTheCorner) {
  SKIP_IF_NO_CUDA_DEVICE();
  {
    const RenderConfig config = makeConfig(400, 400, 60.0f, 0.0f, 20.0f);
    const double rpx = analyticShadowRadiusPx(config);
    const int x = 200 + static_cast<int>(std::lround(1.1 * rpx));
    const Image lensed = blackhole::render(config);
    const Image flat = blackhole::renderFlat(config);
    ASSERT_FALSE(isBlack(lensed, x, 200));
    int maxDiff = 0;
    for (int ch = 0; ch < 3; ++ch) {
      maxDiff = std::max(maxDiff, std::abs(channel(lensed, x, 200, ch) -
                                           channel(flat, x, 200, ch)));
    }
    EXPECT_GE(maxDiff, 30);
    EXPECT_TRUE(isBlack(lensed, 200, 200));
    EXPECT_FALSE(isBlack(flat, 200, 200));  // flat has no shadow
  }
  {
    const RenderConfig config = makeConfig(400, 400, 60.0f, 0.0f, 40.0f);
    const Image lensed = blackhole::render(config);
    const Image flat = blackhole::renderFlat(config);
    for (const auto& xy : {std::pair{0, 0}, std::pair{399, 0},
                           std::pair{0, 399}, std::pair{399, 399}}) {
      for (int ch = 0; ch < 3; ++ch) {
        EXPECT_NEAR(channel(lensed, xy.first, xy.second, ch),
                    channel(flat, xy.first, xy.second, ch), 10)
            << "corner " << xy.first << "," << xy.second << " ch " << ch;
      }
    }
  }
}

TEST(RenderShadowTest, DeterministicAcrossRepeatedRenders) {
  SKIP_IF_NO_CUDA_DEVICE();
  const RenderConfig config = makeConfig(96, 64, 60.0f, 30.0f, 20.0f);
  const Image a = blackhole::render(config);
  const Image b = blackhole::render(config);
  ASSERT_EQ(a.size(), b.size());
  EXPECT_TRUE(std::equal(a.data(), a.data() + a.size(), b.data()));
  EXPECT_GT(blackCount(a), 0u);  // not an all-sky buffer
}

TEST(RenderShadowTest, InvalidConfigThrows) {
  RenderConfig c = makeConfig(8, 8, 60.0f, 90.0f, 20.0f);
  EXPECT_THROW(blackhole::render(c), std::invalid_argument);
  c = makeConfig(0, 8, 60.0f, 0.0f, 20.0f);
  EXPECT_THROW(blackhole::render(c), std::invalid_argument);
  c = makeConfig(8, 8, 180.0f, 0.0f, 20.0f);
  EXPECT_THROW(blackhole::render(c), std::invalid_argument);
  c = makeConfig(8, 8, 60.0f, 0.0f, 0.0f);
  EXPECT_THROW(blackhole::render(c), std::invalid_argument);
}

}  // namespace
