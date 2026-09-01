// GPU tests of the Phase-1 flat-space renderFlat() (Phase 2 made render() lens): every pixel is
// skyGradient(primaryRayDir(camera, px, py)), quantized as
// uint8(round(255 * c)), alpha 255. The expected image is computed here on the
// CPU from camera.hpp (host build of the same functions the kernel uses), and
// a few pixels are additionally pinned to hand-computed constants that do not
// depend on camera.hpp at all.
//
// Tolerance: the device build may differ from the host build by a few ulp
// (FMA contraction, sinf/tanf/rsqrtf implementations). One float ulp near 1.0
// is 6e-8, i.e. 1.5e-5 of a byte step, so a channel can only flip its rounding
// when 255*c lies within ~1e-4 of a .5 boundary: expected for far fewer than 1%
// of channels. We therefore require
//   (a) no channel off by more than 1, and
//   (b) at most 1% of channels off by exactly 1.
// (b) is what keeps the comparison non-vacuous: truncation instead of rounding,
// or a half-pixel sampling offset, shifts ~50% of channels by one.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "blackhole/camera.hpp"
#include "blackhole/image.hpp"
#include "blackhole/render.hpp"
#include "cuda_test_support.hpp"

namespace {

using blackhole::Image;
using blackhole::RenderConfig;
namespace cam = blackhole::cam;

std::uint8_t quantize(float c) {
  return static_cast<std::uint8_t>(std::lround(255.0f * c));
}

// CPU reference: RGBA bytes, row-major, same layout as Image.
std::vector<std::uint8_t> expectedRgba(const RenderConfig& config) {
  const cam::Camera camera =
      cam::makeCamera(config.width, config.height, config.observerDistance,
                      config.fovDegrees, config.elevationDegrees);
  std::vector<std::uint8_t> out(static_cast<std::size_t>(config.width) *
                                config.height * 4);
  for (int y = 0; y < config.height; ++y) {
    for (int x = 0; x < config.width; ++x) {
      const cam::Vec3 c = cam::skyGradient(cam::primaryRayDir(camera, x, y));
      const std::size_t i =
          (static_cast<std::size_t>(y) * config.width + x) * 4;
      out[i + 0] = quantize(c.x);
      out[i + 1] = quantize(c.y);
      out[i + 2] = quantize(c.z);
      out[i + 3] = 255;
    }
  }
  return out;
}

int channel(const Image& image, int x, int y, int ch) {
  return image.data()[image.pixelIndex(x, y) + ch];
}

RenderConfig makeConfig(int w, int h, float fov, float el, float dist = 20.0f) {
  RenderConfig c;
  c.width = w;
  c.height = h;
  c.fovDegrees = fov;
  c.elevationDegrees = el;
  c.observerDistance = dist;
  return c;
}

bool sameBytes(const Image& a, const Image& b) {
  return a.size() == b.size() &&
         std::equal(a.data(), a.data() + a.size(), b.data());
}

struct FlatCase {
  int w, h;
  float fov, el, dist;
};

class RenderFlatMatchesCpuTest : public ::testing::TestWithParam<FlatCase> {};

}  // namespace

TEST_P(RenderFlatMatchesCpuTest, EveryPixelMatchesCpuReference) {
  SKIP_IF_NO_CUDA_DEVICE();
  const FlatCase p = GetParam();
  const RenderConfig config = makeConfig(p.w, p.h, p.fov, p.el, p.dist);

  const Image image = blackhole::renderFlat(config);
  ASSERT_EQ(image.width(), p.w);
  ASSERT_EQ(image.height(), p.h);
  ASSERT_EQ(image.size(), static_cast<std::size_t>(p.w) * p.h * 4);

  const std::vector<std::uint8_t> expected = expectedRgba(config);
  std::size_t off_by_one = 0;
  for (int y = 0; y < p.h; ++y) {
    for (int x = 0; x < p.w; ++x) {
      const std::size_t i = image.pixelIndex(x, y);
      ASSERT_EQ(image.data()[i + 3], 255) << "alpha at " << x << "," << y;
      for (int ch = 0; ch < 3; ++ch) {
        const int diff = std::abs(static_cast<int>(image.data()[i + ch]) -
                                  static_cast<int>(expected[i + ch]));
        ASSERT_LE(diff, 1) << "pixel " << x << "," << y << " channel " << ch
                           << ": got " << int{image.data()[i + ch]}
                           << ", expected " << int{expected[i + ch]};
        off_by_one += static_cast<std::size_t>(diff);
      }
    }
  }
  EXPECT_LE(off_by_one, static_cast<std::size_t>(p.w) * p.h * 3 / 100)
      << "too many channels differ from the CPU reference by 1";
}

INSTANTIATE_TEST_SUITE_P(
    FlatSpace, RenderFlatMatchesCpuTest,
    ::testing::Values(FlatCase{64, 48, 60.0f, 0.0f, 20.0f},
                      FlatCase{64, 48, 60.0f, 30.0f, 20.0f},
                      FlatCase{31, 17, 60.0f, 0.0f, 20.0f},
                      FlatCase{31, 17, 90.0f, 30.0f, 12.0f},
                      FlatCase{17, 31, 45.0f, 60.0f, 30.0f},
                      FlatCase{16, 9, 120.0f, 89.9f, 20.0f}));

// Hand-computed constants, independent of camera.hpp. Image 31x17 (odd, odd):
// the center pixel (15, 8) has ndc = (0, 0) exactly, so its ray is `forward`.
//   el = 0 : forward = (0, 0, 1)            -> color (.5, .5, 1)
//            bytes (round(127.5) = 128, 128, 255); truncation would give 127.
//   el = 30: forward = (0, -.5, cos 30)     -> color (.5, .25, .5 + .25*sqrt3)
//            = (.5, .25, .93301270) -> bytes (128, round(63.75) = 64,
//            round(237.93) = 238).
// Tolerance is 0 on the el=0 bytes and 1 on el=30 (float trig on device).
TEST(RenderFlatTest, CenterPixelMatchesHandComputedColors) {
  SKIP_IF_NO_CUDA_DEVICE();

  const Image flat = blackhole::renderFlat(makeConfig(31, 17, 60.0f, 0.0f));
  EXPECT_EQ(channel(flat, 15, 8, 0), 128);
  EXPECT_EQ(channel(flat, 15, 8, 1), 128);
  EXPECT_EQ(channel(flat, 15, 8, 2), 255);
  EXPECT_EQ(channel(flat, 15, 8, 3), 255);

  const Image tilted = blackhole::renderFlat(makeConfig(31, 17, 60.0f, 30.0f));
  EXPECT_NEAR(channel(tilted, 15, 8, 0), 128, 1);
  EXPECT_NEAR(channel(tilted, 15, 8, 1), 64, 1);
  EXPECT_NEAR(channel(tilted, 15, 8, 2), 238, 1);
  EXPECT_EQ(channel(tilted, 15, 8, 3), 255);
}

// el = 0, 64x48, fov 60: the image must actually be a sky gradient with the
// right orientation, not a constant or a flipped/transposed pattern.
//   R = .5 + .5*dir.x : increases left -> right, mirrored about the center
//   G = .5 + .5*dir.y : decreases top -> bottom (row 0 is the top)
//   dir.z is identical for mirrored pixels, so B matches under mirroring.
TEST(RenderFlatTest, EdgeOnImageHasExpectedOrientationAndSymmetry) {
  SKIP_IF_NO_CUDA_DEVICE();
  const int w = 64;
  const int h = 48;
  const Image image = blackhole::renderFlat(makeConfig(w, h, 60.0f, 0.0f));
  ASSERT_EQ(image.width(), w);
  ASSERT_EQ(image.height(), h);

  // Not constant: wide spread in R (across x) and G (across y). Analytic
  // values at these probes are ~51 / ~204 (R) and ~190 / ~65 (G).
  EXPECT_LT(channel(image, 0, h / 2, 0), 80);
  EXPECT_GT(channel(image, w - 1, h / 2, 0), 175);
  EXPECT_GT(channel(image, w / 2, 0, 1), 175);
  EXPECT_LT(channel(image, w / 2, h - 1, 1), 80);

  // Monotonic trend along the middle row (R) and middle column (G). Equal
  // neighbours are allowed only because quantization may collapse them.
  for (int x = 1; x < w; ++x) {
    EXPECT_GE(channel(image, x, h / 2, 0), channel(image, x - 1, h / 2, 0));
  }
  for (int y = 1; y < h; ++y) {
    EXPECT_LE(channel(image, w / 2, y, 1), channel(image, w / 2, y - 1, 1));
  }

  // Mirror symmetry. R(x) + R(w-1-x) = 255 (+-2: two roundings of an exact
  // complement, plus the device/host slack); same for G across rows; B equal.
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      EXPECT_NEAR(channel(image, x, y, 0) + channel(image, w - 1 - x, y, 0),
                  255, 2);
      EXPECT_NEAR(channel(image, x, y, 2), channel(image, w - 1 - x, y, 2), 1);
      EXPECT_NEAR(channel(image, x, y, 1) + channel(image, x, h - 1 - y, 1),
                  255, 2);
      EXPECT_NEAR(channel(image, x, y, 2), channel(image, x, h - 1 - y, 2), 1);
    }
  }
}

TEST(RenderFlatTest, ElevationChangesTheImageAndTiltsTheSkyDown) {
  SKIP_IF_NO_CUDA_DEVICE();
  const Image flat = blackhole::renderFlat(makeConfig(31, 17, 60.0f, 0.0f));
  const Image tilted = blackhole::renderFlat(makeConfig(31, 17, 60.0f, 30.0f));
  EXPECT_FALSE(sameBytes(flat, tilted));
  // Looking down 30 degrees: the center ray has dir.y = -0.5 -> G ~ 64,
  // versus 128 for the edge-on camera.
  EXPECT_LT(channel(tilted, 15, 8, 1), channel(flat, 15, 8, 1) - 50);
}

TEST(RenderFlatTest, FovChangesTheImage) {
  SKIP_IF_NO_CUDA_DEVICE();
  const Image narrow = blackhole::renderFlat(makeConfig(31, 17, 30.0f, 0.0f));
  const Image wide = blackhole::renderFlat(makeConfig(31, 17, 100.0f, 0.0f));
  // Corner pixel: much further from forward with the wide fov -> smaller B
  // (analytic ~240 vs ~177).
  EXPECT_GT(channel(narrow, 0, 0, 2), channel(wide, 0, 0, 2) + 20);
  // Right edge of the middle row: larger R with the wide fov (~182 vs ~243).
  EXPECT_GT(channel(wide, 30, 8, 0), channel(narrow, 30, 8, 0) + 20);
}

TEST(RenderFlatTest, DeterministicAcrossRepeatedRenders) {
  SKIP_IF_NO_CUDA_DEVICE();
  const RenderConfig config = makeConfig(64, 48, 60.0f, 30.0f);
  const Image a = blackhole::renderFlat(config);
  const Image b = blackhole::renderFlat(config);
  EXPECT_TRUE(sameBytes(a, b));
  // ... and it is not an all-zero / constant buffer.
  EXPECT_NE(*std::min_element(a.data(), a.data() + a.size()),
            *std::max_element(a.data(), a.data() + a.size()));
}

TEST(RenderFlatTest, InvalidElevationThrowsBeforeRendering) {
  RenderConfig c = makeConfig(8, 8, 60.0f, 90.0f);
  EXPECT_THROW(blackhole::renderFlat(c), std::invalid_argument);
  c.elevationDegrees = -1.0f;
  EXPECT_THROW(blackhole::renderFlat(c), std::invalid_argument);
  c.elevationDegrees = std::nanf("");
  EXPECT_THROW(blackhole::renderFlat(c), std::invalid_argument);
}
