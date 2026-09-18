#include "blackhole/post.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace blackhole::post;

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

// Reference values computed in double precision from the formulas in the spec.
constexpr float kAces1 = 0.8037974683544302f;     // f(1) = 2.54/3.16
constexpr float kAces018 = 0.2668989203894968f;   // f(0.18)
constexpr float kAces05 = 0.6163069544364508f;    // f(0.5)
constexpr float kAces2 = 0.914855072463768f;      // f(2)
constexpr float kAces4 = 0.9734171097148379f;     // f(4)

}  // namespace

// ---------------------------------------------------------------- Vec3 ------

TEST(PostVec3Test, Arithmetic) {
  const Vec3 a{1.0f, 2.0f, 3.0f};
  const Vec3 b{0.5f, -1.0f, 4.0f};
  const Vec3 s = a + b;
  const Vec3 d = a - b;
  const Vec3 m = a * 2.0f;
  EXPECT_FLOAT_EQ(s.x, 1.5f);
  EXPECT_FLOAT_EQ(s.y, 1.0f);
  EXPECT_FLOAT_EQ(s.z, 7.0f);
  EXPECT_FLOAT_EQ(d.x, 0.5f);
  EXPECT_FLOAT_EQ(d.y, 3.0f);
  EXPECT_FLOAT_EQ(d.z, -1.0f);
  EXPECT_FLOAT_EQ(m.x, 2.0f);
  EXPECT_FLOAT_EQ(m.y, 4.0f);
  EXPECT_FLOAT_EQ(m.z, 6.0f);
}

// ---------------------------------------------------------------- ACES ------

TEST(PostAcesTest, KnownValues) {
  // Tolerance 1e-6: float evaluation of a rational function (few ulps).
  const Vec3 r = acesFilm({0.0f, 1.0f, 0.18f});
  EXPECT_NEAR(r.x, 0.0f, 1e-6f);
  EXPECT_NEAR(r.y, kAces1, 1e-6f);
  EXPECT_NEAR(r.z, kAces018, 1e-6f);

  const Vec3 q = acesFilm({0.5f, 2.0f, 4.0f});
  EXPECT_NEAR(q.x, kAces05, 1e-6f);
  EXPECT_NEAR(q.y, kAces2, 1e-6f);
  EXPECT_NEAR(q.z, kAces4, 1e-6f);
}

TEST(PostAcesTest, ChannelsAreIndependent) {
  const Vec3 r = acesFilm({4.0f, 0.18f, 1.0f});
  EXPECT_NEAR(r.x, kAces4, 1e-6f);
  EXPECT_NEAR(r.y, kAces018, 1e-6f);
  EXPECT_NEAR(r.z, kAces1, 1e-6f);
}

TEST(PostAcesTest, ClampsToOneForLargeInput) {
  // Unclamped, the fit gives f(10) = 251.3/249.04 = 1.009.
  const Vec3 r = acesFilm({10.0f, 1e4f, 1e6f});
  EXPECT_FLOAT_EQ(r.x, 1.0f);
  EXPECT_FLOAT_EQ(r.y, 1.0f);
  EXPECT_FLOAT_EQ(r.z, 1.0f);
}

TEST(PostAcesTest, InfinityMapsToOneWithoutNaN) {
  const Vec3 r = acesFilm({kInf, kInf, kInf});
  EXPECT_FALSE(std::isnan(r.x));
  EXPECT_FLOAT_EQ(r.x, 1.0f);
  EXPECT_FLOAT_EQ(r.y, 1.0f);
  EXPECT_FLOAT_EQ(r.z, 1.0f);
}

TEST(PostAcesTest, FloatMaxIsFiniteAndClamped) {
  // Without the input clamp to 1e4, FLT_MAX * FLT_MAX overflows to inf/inf.
  const Vec3 hi = acesFilm({FLT_MAX, FLT_MAX, FLT_MAX});
  EXPECT_FALSE(std::isnan(hi.x));
  EXPECT_FLOAT_EQ(hi.x, 1.0f);
  EXPECT_FLOAT_EQ(hi.y, 1.0f);
  EXPECT_FLOAT_EQ(hi.z, 1.0f);
  const Vec3 lo = acesFilm({-FLT_MAX, -FLT_MAX, -FLT_MAX});
  EXPECT_FLOAT_EQ(lo.x, 0.0f);
  EXPECT_FLOAT_EQ(lo.y, 0.0f);
  EXPECT_FLOAT_EQ(lo.z, 0.0f);
}

TEST(PostAcesTest, NegativeInputsMapToZero) {
  const Vec3 r = acesFilm({-1.0f, -1e-3f, -kInf});
  EXPECT_FLOAT_EQ(r.x, 0.0f);
  EXPECT_FLOAT_EQ(r.y, 0.0f);
  EXPECT_FLOAT_EQ(r.z, 0.0f);
}

TEST(PostAcesTest, MonotoneAndBounded) {
  float prev = -1.0f;
  for (int i = -100; i <= 2000; ++i) {
    const float x = 0.01f * static_cast<float>(i);
    const float y = acesFilm({x, x, x}).x;
    ASSERT_GE(y, 0.0f) << x;
    ASSERT_LE(y, 1.0f) << x;
    ASSERT_GE(y, prev) << x;
    prev = y;
  }
}

TEST(PostAcesTest, StrictlyIncreasingBelowSaturation) {
  float prev = -1.0f;
  for (int i = 0; i <= 400; ++i) {
    const float y = acesFilm({0.01f * i, 0.0f, 0.0f}).x;
    ASSERT_GT(y, prev) << i;
    prev = y;
  }
}

// ---------------------------------------------------------------- sRGB ------

TEST(PostSrgbTest, KnownValues) {
  // Doubles from the exact OETF; tolerance 1e-5 is far below the ~6e-3 gap
  // between the exact curve and a pure 2.2 gamma at 0.5.
  EXPECT_NEAR(linearToSrgb(0.0f), 0.0f, 1e-6f);
  EXPECT_NEAR(linearToSrgb(1.0f), 1.0f, 1e-6f);
  EXPECT_NEAR(linearToSrgb(0.5f), 0.7353569830524495f, 1e-5f);
  EXPECT_NEAR(linearToSrgb(0.18f), 0.46135612950044164f, 1e-5f);
  // Linear segment: 12.92 * x (a gamma curve gives 0.0437 at 0.001).
  EXPECT_NEAR(linearToSrgb(0.001f), 0.01292f, 1e-6f);
  EXPECT_NEAR(linearToSrgb(0.0031308f), 0.040449936f, 1e-6f);
}

TEST(PostSrgbTest, ContinuousAtThreshold) {
  const float lo = linearToSrgb(0.0031307f);
  const float hi = linearToSrgb(0.0031309f);
  EXPECT_LT(std::fabs(hi - lo), 1e-4f);
  EXPECT_LT(lo, hi);
}

TEST(PostSrgbTest, ClampsInput) {
  EXPECT_FLOAT_EQ(linearToSrgb(-0.5f), 0.0f);
  EXPECT_FLOAT_EQ(linearToSrgb(-kInf), 0.0f);
  EXPECT_NEAR(linearToSrgb(2.0f), 1.0f, 1e-6f);
  EXPECT_NEAR(linearToSrgb(kInf), 1.0f, 1e-6f);
}

TEST(PostSrgbTest, InverseKnownValues) {
  EXPECT_NEAR(srgbToLinear(0.0f), 0.0f, 1e-7f);
  EXPECT_NEAR(srgbToLinear(1.0f), 1.0f, 1e-6f);
  EXPECT_NEAR(srgbToLinear(0.5f), 0.21404114048223255f, 1e-6f);
  EXPECT_NEAR(srgbToLinear(0.02f), 0.0015479876160990713f, 1e-7f);
}

TEST(PostSrgbTest, InverseClampsInput) {
  EXPECT_FLOAT_EQ(srgbToLinear(-0.5f), 0.0f);
  EXPECT_FLOAT_EQ(srgbToLinear(-kInf), 0.0f);
  EXPECT_NEAR(srgbToLinear(2.0f), 1.0f, 1e-6f);
  EXPECT_NEAR(srgbToLinear(kInf), 1.0f, 1e-6f);
}

TEST(PostSrgbTest, RoundTrip) {
  for (int i = 0; i <= 1000; ++i) {
    const float x = static_cast<float>(i) / 1000.0f;
    EXPECT_NEAR(srgbToLinear(linearToSrgb(x)), x, 1e-5f) << x;
    EXPECT_NEAR(linearToSrgb(srgbToLinear(x)), x, 1e-5f) << x;
  }
  // Around both linear/power breakpoints.
  for (float x : {0.003f, 0.0031308f, 0.0032f, 0.04f, 0.04045f, 0.041f}) {
    EXPECT_NEAR(srgbToLinear(linearToSrgb(x)), x, 1e-5f) << x;
    EXPECT_NEAR(linearToSrgb(srgbToLinear(x)), x, 1e-5f) << x;
  }
}

// ---------------------------------------------------------------- bytes -----

TEST(PostByteTest, EndpointsAndRounding) {
  EXPECT_EQ(toByte(0.0f), 0);
  EXPECT_EQ(toByte(1.0f), 255);
  EXPECT_EQ(toByte(0.5f), 128);     // 127.5 is an exact tie -> lround: 128
  EXPECT_EQ(toByte(0.498f), 127);   // 126.99
  EXPECT_EQ(toByte(0.002f), 1);     // 0.51
  EXPECT_EQ(toByte(0.0019f), 0);    // 0.4845
  EXPECT_EQ(toByte(0.25f), 64);     // 63.75
}

TEST(PostByteTest, Clamps) {
  EXPECT_EQ(toByte(-1.0f), 0);
  EXPECT_EQ(toByte(-kInf), 0);
  EXPECT_EQ(toByte(1.5f), 255);
  EXPECT_EQ(toByte(1e30f), 255);
  EXPECT_EQ(toByte(kInf), 255);
}

TEST(PostByteTest, ByteOverBytesRoundTrips) {
  for (int k = 0; k <= 255; ++k) {
    EXPECT_EQ(toByte(static_cast<float>(k) / 255.0f), k) << k;
  }
}

// ---------------------------------------------------------------- pipeline --

TEST(PostPipelineTest, WhiteAtUnitExposure) {
  const Vec3 p = postProcessPixel({1.0f, 1.0f, 1.0f}, 1.0f);
  // ACES(1) = 0.80380 -> sRGB 0.90823 -> 231.6 -> 232.
  EXPECT_NEAR(p.x, 0.9082304956696972f, 1e-5f);
  EXPECT_NEAR(p.y, 0.9082304956696972f, 1e-5f);
  EXPECT_NEAR(p.z, 0.9082304956696972f, 1e-5f);
  EXPECT_EQ(toByte(p.x), 232);  // a 2.2 gamma would give 231
}

TEST(PostPipelineTest, MixedChannelsBytes) {
  std::uint8_t out[4] = {};
  packRgba8(postProcessPixel({0.18f, 1.0f, 4.0f}, 1.0f), out);
  EXPECT_EQ(out[0], 141);  // 0.55346 * 255 = 141.1
  EXPECT_EQ(out[1], 232);
  EXPECT_EQ(out[2], 252);  // 0.98822 * 255 = 252.0
  EXPECT_EQ(out[3], 255);
}

TEST(PostPipelineTest, ExposureAppliedBeforeToneMap) {
  const Vec3 c{0.5f, 0.25f, 1.0f};
  const Vec3 a = postProcessPixel(c, 2.0f);
  const Vec3 t = acesFilm(c * 2.0f);
  EXPECT_NEAR(a.x, linearToSrgb(t.x), 1e-6f);
  EXPECT_NEAR(a.y, linearToSrgb(t.y), 1e-6f);
  EXPECT_NEAR(a.z, linearToSrgb(t.z), 1e-6f);
  // linear 1 at exposure 2 -> ACES(2) = 0.91486 -> sRGB 0.9616 -> 245.
  EXPECT_EQ(toByte(postProcessPixel({1.0f, 1.0f, 1.0f}, 2.0f).x), 245);
}

TEST(PostPipelineTest, ZeroAndNegativeAndHugeInputs) {
  Vec3 z = postProcessPixel({0.0f, 0.0f, 0.0f}, 1.0f);
  EXPECT_FLOAT_EQ(z.x, 0.0f);
  Vec3 n = postProcessPixel({-3.0f, -0.1f, 0.0f}, 1.0f);
  EXPECT_FLOAT_EQ(n.x, 0.0f);
  EXPECT_FLOAT_EQ(n.y, 0.0f);
  Vec3 big = postProcessPixel({1e6f, kInf, 50.0f}, 1.0f);
  EXPECT_NEAR(big.x, 1.0f, 1e-6f);
  EXPECT_NEAR(big.y, 1.0f, 1e-6f);
  EXPECT_NEAR(big.z, 1.0f, 1e-6f);
  Vec3 e0 = postProcessPixel({5.0f, 5.0f, 5.0f}, 0.0f);
  EXPECT_FLOAT_EQ(e0.x, 0.0f);
}

TEST(PostPipelineTest, ExposureMonotone) {
  const float exposures[] = {0.125f, 0.25f, 0.5f, 1.0f, 2.0f, 4.0f};
  float prev = -1.0f;
  int prevByte = -1;
  for (float e : exposures) {
    const Vec3 p = postProcessPixel({0.2f, 0.2f, 0.2f}, e);
    EXPECT_GT(p.x, prev) << e;
    EXPECT_GE(p.x, 0.0f);
    EXPECT_LE(p.x, 1.0f);
    EXPECT_GT(static_cast<int>(toByte(p.x)), prevByte) << e;
    prev = p.x;
    prevByte = toByte(p.x);
  }
}

TEST(PostPipelineTest, RampHasNoBanding) {
  // x = 0.002 j for j in [0, 1000].  Analytically 193 distinct bytes, largest
  // step 3, first 0 and last 245.
  std::vector<int> bytes;
  for (int j = 0; j <= 1000; ++j) {
    const float x = 0.002f * static_cast<float>(j);
    bytes.push_back(toByte(postProcessPixel({x, x, x}, 1.0f).x));
  }
  int maxJump = 0;
  for (std::size_t i = 1; i < bytes.size(); ++i) {
    ASSERT_GE(bytes[i], bytes[i - 1]) << i;
    maxJump = std::max(maxJump, bytes[i] - bytes[i - 1]);
  }
  EXPECT_EQ(bytes.front(), 0);
  EXPECT_NEAR(bytes.back(), 245, 1);
  EXPECT_LE(maxJump, 4);
  const std::set<int> distinct(bytes.begin(), bytes.end());
  EXPECT_GE(distinct.size(), 185u);
}

TEST(PostPipelineTest, ShadowsAreDistinct) {
  // Slope in the shadows is > 150 bytes per unit, so steps of 0.01 up to 0.3
  // must give strictly increasing bytes.
  int prev = -1;
  for (int j = 0; j <= 30; ++j) {
    const float x = 0.01f * static_cast<float>(j);
    const int b = toByte(postProcessPixel({x, x, x}, 1.0f).x);
    EXPECT_GT(b, prev) << j;
    prev = b;
  }
}

TEST(PostPipelineTest, PackRgba8) {
  std::uint8_t out[8];
  std::fill(out, out + 8, std::uint8_t{0xAB});
  packRgba8({0.0f, 0.5f, 1.0f}, out);
  EXPECT_EQ(out[0], 0);
  EXPECT_EQ(out[1], 128);
  EXPECT_EQ(out[2], 255);
  EXPECT_EQ(out[3], 255);
  for (int i = 4; i < 8; ++i) EXPECT_EQ(out[i], 0xAB) << i;

  packRgba8({-1.0f, 2.0f, 0.25f}, out);
  EXPECT_EQ(out[0], 0);
  EXPECT_EQ(out[1], 255);
  EXPECT_EQ(out[2], 64);
  EXPECT_EQ(out[3], 255);
}

// ---------------------------------------------------------------- sampling --

namespace {

int RootOf(int spp) {
  int n = 1;
  while (n * n < spp) ++n;
  return n;
}

bool InStratum(float v, int cell, int n) {
  const float eps = 1e-6f;
  return v >= 0.0f && v < 1.0f && v >= static_cast<float>(cell) / n - eps &&
         v < static_cast<float>(cell + 1) / n + eps;
}

struct EdgeStats {
  double rms, bias, maxAbs;
};

// Pixels sampled with a step edge: value 1 where dx < 0.37.  Analytic
// coverage is 0.37.
EdgeStats StepEdge(int spp, int w, int h) {
  double sum = 0, sumSq = 0, maxAbs = 0;
  for (int py = 0; py < h; ++py) {
    for (int px = 0; px < w; ++px) {
      int hit = 0;
      for (int i = 0; i < spp; ++i) {
        hit += sampleOffset(i, spp, px, py).dx < 0.37f ? 1 : 0;
      }
      const double e = static_cast<double>(hit) / spp - 0.37;
      sum += e;
      sumSq += e * e;
      maxAbs = std::max(maxAbs, std::fabs(e));
    }
  }
  const double n = static_cast<double>(w) * h;
  return {std::sqrt(sumSq / n), sum / n, maxAbs};
}

}  // namespace

TEST(PostSamplingTest, IsValidSpp) {
  for (int spp : {1, 4, 9, 16, 25, 36, 64, 100, 4096 * 4096, 46340 * 46340}) {
    EXPECT_TRUE(isValidSpp(spp)) << spp;
  }
  for (int spp : {0, -1, -4, 2, 3, 5, 8, 15, 17, 63, 65, 46340 * 46340 - 1,
                  46340 * 46340 + 1}) {
    EXPECT_FALSE(isValidSpp(spp)) << spp;
  }
}

TEST(PostSamplingTest, SingleSampleIsPixelCentre) {
  for (int py = 0; py < 20; ++py) {
    for (int px = 0; px < 20; ++px) {
      const SampleOffset o = sampleOffset(0, 1, px * 37, py * 11);
      EXPECT_EQ(o.dx, 0.5f);
      EXPECT_EQ(o.dy, 0.5f);
    }
  }
}

TEST(PostSamplingTest, EachCellHasExactlyItsSample) {
  for (int spp : {1, 4, 9, 16, 25, 36, 64}) {
    const int n = RootOf(spp);
    for (int k = 0; k < 200; ++k) {
      const int px = (k * 7919) % 8192;
      const int py = (k * 104729) % 4320;
      for (int i = 0; i < spp; ++i) {
        const SampleOffset o = sampleOffset(i, spp, px, py);
        ASSERT_TRUE(InStratum(o.dx, i % n, n))
            << "spp " << spp << " i " << i << " dx " << o.dx;
        ASSERT_TRUE(InStratum(o.dy, i / n, n))
            << "spp " << spp << " i " << i << " dy " << o.dy;
      }
    }
  }
}

TEST(PostSamplingTest, OffsetsInUnitIntervalForLargeCoordinates) {
  for (int spp : {1, 4, 16, 64}) {
    for (int k = 0; k < 500; ++k) {
      const int px = 100000 + k * 31;
      const int py = 7 * k;
      for (int i = 0; i < spp; ++i) {
        const SampleOffset o = sampleOffset(i, spp, px, py);
        ASSERT_GE(o.dx, 0.0f);
        ASSERT_LT(o.dx, 1.0f);
        ASSERT_GE(o.dy, 0.0f);
        ASSERT_LT(o.dy, 1.0f);
      }
    }
  }
}

TEST(PostSamplingTest, MaximumSppLastCellsStayBelowOne) {
  // spp = 46340^2 = 2147395600 (largest valid int square).  Last row / last
  // column cells have (c+u)/n within float rounding of 1.0; offsets must
  // still be in [0, 1).  n*n must not overflow 32-bit int in the callee.
  const int n = 46340;
  const int spp = n * n;
  ASSERT_TRUE(isValidSpp(spp));
  const int cells[] = {spp - 1, spp - 2, spp - n, n - 1, n, 0, spp - n - 1};
  for (int k = 0; k < 4000; ++k) {
    const int px = (k * 7919) % 8192;
    const int py = (k * 104729) % 4320;
    for (int i : cells) {
      const long long cx = static_cast<long long>(i) % n;
      const long long cy = static_cast<long long>(i) / n;
      const SampleOffset o = sampleOffset(i, spp, px, py);
      ASSERT_GE(o.dx, 0.0f) << i;
      ASSERT_LT(o.dx, 1.0f) << i;
      ASSERT_GE(o.dy, 0.0f) << i;
      ASSERT_LT(o.dy, 1.0f) << i;
      // Coarse stratum check (float resolution near n is ~4e-3 in cell units).
      ASSERT_GE(o.dx, static_cast<float>(cx) / n - 1e-6f) << i;
      ASSERT_GE(o.dy, static_cast<float>(cy) / n - 1e-6f) << i;
    }
  }
}

TEST(PostSamplingTest, Deterministic) {
  std::vector<SampleOffset> fwd, rev;
  for (int px = 0; px < 16; ++px)
    for (int i = 0; i < 16; ++i) fwd.push_back(sampleOffset(i, 16, px, 3));
  for (int px = 15; px >= 0; --px)
    for (int i = 15; i >= 0; --i) rev.push_back(sampleOffset(i, 16, px, 3));
  std::reverse(rev.begin(), rev.end());
  ASSERT_EQ(fwd.size(), rev.size());
  for (std::size_t k = 0; k < fwd.size(); ++k) {
    EXPECT_EQ(fwd[k].dx, rev[k].dx);
    EXPECT_EQ(fwd[k].dy, rev[k].dy);
  }
}

TEST(PostSamplingTest, PatternVariesBetweenPixels) {
  // Sample 0 of spp=4 over a 32x32 patch: nearly all offsets distinct.
  std::set<std::pair<float, float>> seen;
  for (int py = 0; py < 32; ++py)
    for (int px = 0; px < 32; ++px) {
      const SampleOffset o = sampleOffset(0, 4, px, py);
      seen.insert({o.dx, o.dy});
    }
  EXPECT_GE(seen.size(), 1000u);

  int diffX = 0, diffY = 0, diffSwap = 0, total = 0, swapTotal = 0;
  for (int py = 0; py < 31; ++py)
    for (int px = 0; px < 31; ++px) {
      const SampleOffset a = sampleOffset(0, 4, px, py);
      const SampleOffset r = sampleOffset(0, 4, px + 1, py);
      const SampleOffset d = sampleOffset(0, 4, px, py + 1);
      diffX += (a.dx != r.dx || a.dy != r.dy);
      diffY += (a.dx != d.dx || a.dy != d.dy);
      ++total;
      if (px != py) {
        const SampleOffset s = sampleOffset(0, 4, py, px);
        diffSwap += (a.dx != s.dx || a.dy != s.dy);
        ++swapTotal;
      }
    }
  EXPECT_GE(diffX, total * 95 / 100);
  EXPECT_GE(diffY, total * 95 / 100);
  EXPECT_GE(diffSwap, swapTotal * 95 / 100);
}

TEST(PostSamplingTest, JitterVariesWithinPixel) {
  // The jitter fractions of the 16 cells of one pixel must not be identical.
  std::set<float> us;
  for (int i = 0; i < 16; ++i) {
    const SampleOffset o = sampleOffset(i, 16, 5, 9);
    us.insert(o.dx * 4.0f - static_cast<float>(i % 4));
  }
  EXPECT_GE(us.size(), 12u);
}

TEST(PostSamplingTest, JitterIsUniformAndIndependentAcrossAxes) {
  // 4096 pixels x 16 samples = 65536 jitter fractions u in [0,1).
  const int n = 4;
  std::vector<double> ux, uy;
  for (int py = 0; py < 64; ++py)
    for (int px = 0; px < 64; ++px)
      for (int i = 0; i < 16; ++i) {
        const SampleOffset o = sampleOffset(i, 16, px, py);
        ux.push_back(o.dx * n - (i % n));
        uy.push_back(o.dy * n - (i / n));
      }
  const double N = static_cast<double>(ux.size());
  double mx = 0, my = 0, mn = 1, mxv = 0;
  int bins[8] = {};
  for (std::size_t k = 0; k < ux.size(); ++k) {
    mx += ux[k];
    my += uy[k];
    mn = std::min(mn, ux[k]);
    mxv = std::max(mxv, ux[k]);
    ++bins[std::min(7, std::max(0, static_cast<int>(ux[k] * 8)))];
  }
  mx /= N;
  my /= N;
  // sd of the mean = 0.289/sqrt(65536) = 1.1e-3; tolerance is ~9 sd.
  EXPECT_NEAR(mx, 0.5, 0.01);
  EXPECT_NEAR(my, 0.5, 0.01);
  EXPECT_LT(mn, 0.002);   // reaches the lower end of the stratum
  EXPECT_GT(mxv, 0.998);  // and the upper end
  // Expected 8192 per bin, sd ~85; tolerance 6% = 491 (5.7 sd).
  for (int b = 0; b < 8; ++b) EXPECT_NEAR(bins[b], 8192, 491) << b;
  // Correlation between axes: sd 0.0039; tolerance 0.05.
  double cov = 0;
  for (std::size_t k = 0; k < ux.size(); ++k)
    cov += (ux[k] - mx) * (uy[k] - my);
  EXPECT_LT(std::fabs(cov / N / (1.0 / 12.0)), 0.05);
}

TEST(PostSamplingTest, MeanOffsetPerPixelNearHalf) {
  for (int spp : {16, 25, 64}) {
    double gx = 0, gy = 0;
    const int w = 32, h = 32;
    for (int py = 0; py < h; ++py)
      for (int px = 0; px < w; ++px) {
        double sx = 0, sy = 0;
        for (int i = 0; i < spp; ++i) {
          const SampleOffset o = sampleOffset(i, spp, px, py);
          sx += o.dx;
          sy += o.dy;
        }
        sx /= spp;
        sy /= spp;
        // Stratification pins the mean to 0.5 +- jitter noise (sd < 0.02).
        ASSERT_NEAR(sx, 0.5, 0.1) << spp;
        ASSERT_NEAR(sy, 0.5, 0.1) << spp;
        gx += sx;
        gy += sy;
      }
    EXPECT_NEAR(gx / (w * h), 0.5, 0.01) << spp;
    EXPECT_NEAR(gy / (w * h), 0.5, 0.01) << spp;
  }
}

TEST(PostSamplingTest, StepEdgeErrorShrinksWithSpp) {
  // For n x n stratified samples only the column containing the edge is
  // uncertain, so |error| <= 1/n per pixel and the rms error is
  // sqrt(p(1-p)) n^-1.5 with p = frac(0.37 n):
  //   spp 1: 0.37 (centre sample reads 0 everywhere), spp 4: 0.156,
  //   spp 16: 0.0625, spp 64: 0.0087.
  const int w = 64, h = 64;
  const EdgeStats s1 = StepEdge(1, w, h);
  const EdgeStats s4 = StepEdge(4, w, h);
  const EdgeStats s16 = StepEdge(16, w, h);
  const EdgeStats s64 = StepEdge(64, w, h);

  EXPECT_NEAR(s1.rms, 0.37, 1e-6);
  EXPECT_NEAR(s1.maxAbs, 0.37, 1e-6);

  // Deterministic per-pixel bound of a correct stratified estimator.  Full
  // random (unstratified) jitter would exceed 0.25 at spp 16 for some pixel.
  EXPECT_LE(s4.maxAbs, 0.5 + 1e-6);
  EXPECT_LE(s16.maxAbs, 0.25 + 1e-6);
  EXPECT_LE(s64.maxAbs, 0.125 + 1e-6);
  // Worst case at spp 16 is already better than the centre sample's 0.37.
  EXPECT_LT(s16.maxAbs, s1.maxAbs);

  // Each step reduces rms by >= 40% (expected ratios 0.42, 0.40, 0.14).
  EXPECT_LT(s4.rms, 0.6 * s1.rms);
  EXPECT_LT(s16.rms, 0.6 * s4.rms);
  EXPECT_LT(s64.rms, 0.6 * s16.rms);
  EXPECT_LE(s4.rms, 0.2);
  EXPECT_LE(s16.rms, 0.08);   // a jitter-free midpoint grid gives 0.12
  EXPECT_LE(s64.rms, 0.012);

  // Unbiased: sd of the mean is rms/64; tolerances are >= 5 sd.
  EXPECT_LT(std::fabs(s4.bias), 0.015);
  EXPECT_LT(std::fabs(s16.bias), 0.005);
  EXPECT_LT(std::fabs(s64.bias), 0.001);
}

// ---------------------------------------------------------------- bloom -----

TEST(PostBloomKernelTest, Radius) {
  EXPECT_EQ(gaussianKernelRadius(1.0f), 3);
  EXPECT_EQ(gaussianKernelRadius(0.5f), 2);
  EXPECT_EQ(gaussianKernelRadius(2.5f), 8);
  EXPECT_EQ(gaussianKernelRadius(4.0f), 12);
  EXPECT_EQ(gaussianKernelRadius(10.2f), 31);
  EXPECT_EQ(gaussianKernelRadius(0.1f), 1);
  EXPECT_EQ(gaussianKernelRadius(0.01f), 1);
}

TEST(PostBloomKernelTest, SizeSymmetryNormalisationPeak) {
  for (float sigma : {0.1f, 0.5f, 1.0f, 2.0f, 3.7f, 8.0f}) {
    const std::vector<float> k = gaussianKernel1D(sigma);
    const int r = gaussianKernelRadius(sigma);
    ASSERT_EQ(k.size(), static_cast<std::size_t>(2 * r + 1)) << sigma;
    double sum = 0;
    for (float w : k) sum += w;
    EXPECT_NEAR(sum, 1.0, 1e-6) << sigma;
    for (int i = 0; i <= r; ++i) {
      EXPECT_FLOAT_EQ(k[r - i], k[r + i]) << sigma << " " << i;
    }
    for (int i = 1; i <= r; ++i) {
      EXPECT_GE(k[r + i - 1], k[r + i]) << sigma << " " << i;
      EXPECT_GT(k[r], k[r + i]) << sigma << " " << i;
    }
  }
}

TEST(PostBloomKernelTest, KnownWeightsSigma1) {
  // exp(-x^2/2) for x = 0..3, divided by their symmetric sum.
  const std::vector<float> k = gaussianKernel1D(1.0f);
  const float expected[] = {0.004433048f, 0.054005583f, 0.24203623f,
                            0.39905028f};
  ASSERT_EQ(k.size(), 7u);
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(k[3 + i], expected[3 - i], 1e-6f) << i;
    EXPECT_NEAR(k[3 - i], expected[3 - i], 1e-6f) << i;
  }
}

TEST(PostBloomKernelTest, KnownWeightsSigmaHalf) {
  const std::vector<float> k = gaussianKernel1D(0.5f);
  ASSERT_EQ(k.size(), 5u);
  EXPECT_NEAR(k[2], 0.78657073f, 1e-6f);
  EXPECT_NEAR(k[3], 0.10645078f, 1e-6f);
  EXPECT_NEAR(k[4], 0.00026386509f, 1e-6f);
}

TEST(PostBloomKernelTest, InvalidSigmaThrows) {
  EXPECT_THROW(gaussianKernel1D(0.0f), std::invalid_argument);
  EXPECT_THROW(gaussianKernel1D(-1.0f), std::invalid_argument);
  EXPECT_THROW(gaussianKernel1D(kInf), std::invalid_argument);
  EXPECT_THROW(gaussianKernel1D(-kInf), std::invalid_argument);
  EXPECT_THROW(gaussianKernel1D(std::numeric_limits<float>::quiet_NaN()),
               std::invalid_argument);
  EXPECT_NO_THROW(gaussianKernel1D(1.0f));
}

TEST(PostBlurRowTest, ConstantRowStaysConstant) {
  const std::vector<float> k = gaussianKernel1D(2.0f);  // radius 6
  for (std::size_t n : {std::size_t{3}, std::size_t{20}}) {  // 3 < radius
    const std::vector<float> row(n, 0.75f);
    const std::vector<float> out = blurRow(row, k);
    ASSERT_EQ(out.size(), n);
    for (std::size_t i = 0; i < n; ++i) EXPECT_NEAR(out[i], 0.75f, 1e-6f) << i;
  }
}

TEST(PostBlurRowTest, DeltaProducesKernel) {
  const std::vector<float> k = gaussianKernel1D(1.0f);  // radius 3
  std::vector<float> row(31, 0.0f);
  row[15] = 1.0f;
  const std::vector<float> out = blurRow(row, k);
  ASSERT_EQ(out.size(), row.size());
  for (std::size_t i = 0; i < out.size(); ++i) {
    const int j = static_cast<int>(i) - 15 + 3;
    const float want = (j >= 0 && j < 7) ? k[j] : 0.0f;
    EXPECT_NEAR(out[i], want, 1e-7f) << i;
  }
}

TEST(PostBlurRowTest, ClampToEdgeAtLeftBoundary) {
  // Delta at index 0, sigma 1: out[i] sums kernel taps that read index 0
  // after clamping.  Zero padding or mirroring would give 0.3991 at i=0.
  const std::vector<float> k = gaussianKernel1D(1.0f);
  std::vector<float> row(8, 0.0f);
  row[0] = 1.0f;
  const std::vector<float> out = blurRow(row, k);
  EXPECT_NEAR(out[0], 0.69952514f, 1e-6f);  // K0+K1+K2+K3
  EXPECT_NEAR(out[1], 0.30047486f, 1e-6f);  // K0+K1+K2
  EXPECT_NEAR(out[2], 0.05843863f, 1e-6f);  // K0+K1
  EXPECT_NEAR(out[3], 0.004433048f, 1e-6f); // K0
  EXPECT_NEAR(out[4], 0.0f, 1e-7f);
}

TEST(PostBlurRowTest, ClampToEdgeAtRightBoundary) {
  const std::vector<float> k = gaussianKernel1D(1.0f);
  std::vector<float> row(8, 0.0f);
  row[7] = 1.0f;
  const std::vector<float> out = blurRow(row, k);
  EXPECT_NEAR(out[7], 0.69952514f, 1e-6f);
  EXPECT_NEAR(out[6], 0.30047486f, 1e-6f);
  EXPECT_NEAR(out[5], 0.05843863f, 1e-6f);
  EXPECT_NEAR(out[4], 0.004433048f, 1e-6f);
  EXPECT_NEAR(out[3], 0.0f, 1e-7f);
}

TEST(PostBlurRowTest, EnergyConservedAwayFromEdges) {
  const std::vector<float> k = gaussianKernel1D(1.0f);  // radius 3
  std::vector<float> row(64, 0.0f);
  double in = 0;
  for (int i = 20; i < 44; ++i) {
    row[i] = static_cast<float>((i * 37) % 11) / 10.0f;
    in += row[i];
  }
  const std::vector<float> out = blurRow(row, k);
  double sum = 0;
  for (float v : out) sum += v;
  EXPECT_NEAR(sum, in, 1e-4);
}

TEST(PostBlurRowTest, LinearRampPreservedInInterior) {
  const std::vector<float> k = gaussianKernel1D(2.0f);  // radius 6
  std::vector<float> row(40);
  for (int i = 0; i < 40; ++i) row[i] = static_cast<float>(i);
  const std::vector<float> out = blurRow(row, k);
  for (int i = 6; i < 34; ++i) EXPECT_NEAR(out[i], static_cast<float>(i), 1e-4f);
}

TEST(PostBloomCombineTest, AddsScaledBlur) {
  const Vec3 r = bloomCombine({1.0f, 2.0f, 3.0f}, {2.0f, 4.0f, 6.0f}, 0.5f);
  EXPECT_FLOAT_EQ(r.x, 2.0f);
  EXPECT_FLOAT_EQ(r.y, 4.0f);
  EXPECT_FLOAT_EQ(r.z, 6.0f);
  const Vec3 z = bloomCombine({1.0f, 2.0f, 3.0f}, {9.0f, 9.0f, 9.0f}, 0.0f);
  EXPECT_FLOAT_EQ(z.x, 1.0f);
  EXPECT_FLOAT_EQ(z.z, 3.0f);
  const Vec3 s = bloomCombine({0.0f, 0.0f, 0.0f}, {1.0f, 2.0f, 3.0f}, 2.0f);
  EXPECT_FLOAT_EQ(s.y, 4.0f);
}
