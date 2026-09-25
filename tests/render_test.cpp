#include "blackhole/render.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <string>

#include "cuda_test_support.hpp"

using blackhole::RenderConfig;

TEST(RenderConfigTest, ValidateAcceptsDefaults) {
  EXPECT_NO_THROW(blackhole::validate(RenderConfig{}));
}

TEST(RenderConfigTest, DefaultElevationIsTenDegrees) {
  EXPECT_FLOAT_EQ(RenderConfig{}.elevationDegrees, 10.0f);
}

TEST(RenderConfigTest, ValidateAcceptsElevationInHalfOpenRange) {
  RenderConfig c;
  for (const float el : {0.0f, 10.0f, 45.0f, 89.9f}) {
    c.elevationDegrees = el;
    EXPECT_NO_THROW(blackhole::validate(c)) << "elevation " << el;
  }
}

TEST(RenderConfigTest, ValidateRejectsElevationOutsideRange) {
  RenderConfig c;
  for (const float el :
       {-1.0f, -0.001f, 90.0f, 180.0f, std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()}) {
    c.elevationDegrees = el;
    try {
      blackhole::validate(c);
      ADD_FAILURE() << "expected std::invalid_argument for elevation " << el;
    } catch (const std::invalid_argument &e) {
      std::string msg = e.what();
      std::transform(msg.begin(), msg.end(), msg.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
      });
      EXPECT_NE(msg.find("elevation"), std::string::npos)
          << "elevation " << el << ": " << msg;
    }
  }
}

TEST(RenderConfigTest, ValidateRejectsBadValues) {
  RenderConfig c;
  c.width = 0;
  EXPECT_THROW(blackhole::validate(c), std::invalid_argument);

  c = RenderConfig{};
  c.height = -1;
  EXPECT_THROW(blackhole::validate(c), std::invalid_argument);

  c = RenderConfig{};
  c.observerDistance = 0.0f;
  EXPECT_THROW(blackhole::validate(c), std::invalid_argument);

  c = RenderConfig{};
  c.fovDegrees = 180.0f;
  EXPECT_THROW(blackhole::validate(c), std::invalid_argument);
}

namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr float kInf = std::numeric_limits<float>::infinity();

std::string lower(std::string msg) {
  std::transform(msg.begin(), msg.end(), msg.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return msg;
}

// Expects validate(c) to throw invalid_argument whose message names `field`.
void expectRejected(const RenderConfig &c, const std::string &field) {
  try {
    blackhole::validate(c);
    ADD_FAILURE() << "expected std::invalid_argument for " << field;
  } catch (const std::invalid_argument &e) {
    EXPECT_NE(lower(e.what()).find(lower(field)), std::string::npos)
        << field << ": " << e.what();
  }
}

} // namespace

TEST(RenderConfigTest, WiringDefaultsPreserveCurrentBehavior) {
  const RenderConfig c;
  EXPECT_EQ(c.sky, blackhole::SkyMode::Gradient);
  EXPECT_TRUE(c.skyImagePath.empty());
  EXPECT_FALSE(c.disk);
  EXPECT_FLOAT_EQ(c.diskInner, 3.0f);
  EXPECT_FLOAT_EQ(c.diskOuter, 12.0f);
  EXPECT_FLOAT_EQ(c.diskTempK, 12000.0f);
  EXPECT_FLOAT_EQ(c.diskBrightness, 1.0f);
  EXPECT_FALSE(c.postProcess);
  EXPECT_FLOAT_EQ(c.exposure, 1.0f);
  EXPECT_EQ(c.spp, 1);
  EXPECT_FLOAT_EQ(c.bloomStrength, 0.0f);
}

TEST(RenderConfigTest, ValidateDiskInner) {
  RenderConfig c;
  c.diskInner = 1.5f;
  EXPECT_NO_THROW(blackhole::validate(c));
  c.diskInner = 11.9f;
  EXPECT_NO_THROW(blackhole::validate(c));
  for (const float v : {1.0f, 0.5f, -3.0f, kNaN, kInf, -kInf}) {
    c = RenderConfig{};
    c.diskInner = v;
    expectRejected(c, "diskInner");
  }
}

TEST(RenderConfigTest, ValidateDiskOuter) {
  RenderConfig c;
  c.diskOuter = 3.5f;
  EXPECT_NO_THROW(blackhole::validate(c));
  c.diskOuter = 1000.0f;
  EXPECT_NO_THROW(blackhole::validate(c));
  for (const float v : {3.0f, 2.0f, kNaN, kInf, -kInf}) {
    c = RenderConfig{};
    c.diskOuter = v;
    expectRejected(c, "diskOuter");
  }
}

TEST(RenderConfigTest, ValidateDiskTemp) {
  RenderConfig c;
  for (const float v : {1000.0f, 5000.0f, 40000.0f}) {
    c.diskTempK = v;
    EXPECT_NO_THROW(blackhole::validate(c)) << v;
  }
  for (const float v : {999.9f, 40000.1f, 0.0f, -1.0f, kNaN, kInf, -kInf}) {
    c = RenderConfig{};
    c.diskTempK = v;
    expectRejected(c, "diskTempK");
  }
}

TEST(RenderConfigTest, ValidateDiskBrightness) {
  RenderConfig c;
  for (const float v : {0.0f, 0.5f, 100.0f}) {
    c.diskBrightness = v;
    EXPECT_NO_THROW(blackhole::validate(c)) << v;
  }
  for (const float v : {-0.001f, kNaN, kInf, -kInf}) {
    c = RenderConfig{};
    c.diskBrightness = v;
    expectRejected(c, "diskBrightness");
  }
}

TEST(RenderConfigTest, ValidateExposure) {
  RenderConfig c;
  for (const float v : {0.001f, 1.0f, 50.0f}) {
    c.exposure = v;
    EXPECT_NO_THROW(blackhole::validate(c)) << v;
  }
  for (const float v : {0.0f, -1.0f, kNaN, kInf, -kInf}) {
    c = RenderConfig{};
    c.exposure = v;
    expectRejected(c, "exposure");
  }
}

TEST(RenderConfigTest, ValidateBloomStrength) {
  RenderConfig c;
  for (const float v : {0.0f, 0.3f, 10.0f}) {
    c.bloomStrength = v;
    EXPECT_NO_THROW(blackhole::validate(c)) << v;
  }
  for (const float v : {-0.001f, kNaN, kInf, -kInf}) {
    c = RenderConfig{};
    c.bloomStrength = v;
    expectRejected(c, "bloomStrength");
  }
}

TEST(RenderConfigTest, ValidateSppPerfectSquaresOnly) {
  RenderConfig c;
  for (const int v : {1, 4, 9, 16, 25, 36, 49, 64}) {
    c.spp = v;
    EXPECT_NO_THROW(blackhole::validate(c)) << v;
  }
  for (const int v : {0, -1, -4, 2, 3, 5, 8, 10, 63, 65, 81, 100}) {
    c = RenderConfig{};
    c.spp = v;
    expectRejected(c, "spp");
  }
}

TEST(RenderConfigTest, ValidateSkyImageNeedsPath) {
  RenderConfig c;
  c.sky = blackhole::SkyMode::Image;
  expectRejected(c, "skyImagePath");
  c.skyImagePath = "sky.hdr";
  EXPECT_NO_THROW(blackhole::validate(c));
}

TEST(RenderConfigTest, SkyImagePathIgnoredUnlessImageMode) {
  RenderConfig c;
  c.skyImagePath = "ignored.png";
  for (const auto mode :
       {blackhole::SkyMode::Gradient, blackhole::SkyMode::Starfield}) {
    c.sky = mode;
    EXPECT_NO_THROW(blackhole::validate(c));
  }
  c.sky = blackhole::SkyMode::Starfield;
  c.skyImagePath.clear();
  EXPECT_NO_THROW(blackhole::validate(c));
}

TEST(RenderTest, RejectsInvalidWiringConfigBeforeTouchingTheGpu) {
  RenderConfig c;
  c.spp = 2;
  EXPECT_THROW(blackhole::render(c), std::invalid_argument);
  EXPECT_THROW(blackhole::renderFlat(c), std::invalid_argument);
}

TEST(RenderTest, RejectsInvalidConfigBeforeTouchingTheGpu) {
  RenderConfig c;
  c.width = 0;
  EXPECT_THROW(blackhole::render(c), std::invalid_argument);
}

TEST(RenderTest, RenderFlatRejectsInvalidConfigBeforeTouchingTheGpu) {
  RenderConfig c;
  c.width = 0;
  EXPECT_THROW(blackhole::renderFlat(c), std::invalid_argument);
}

// Smoke test of the public entry point the CLI uses.
TEST(RenderTest, ReturnsImageWithRequestedSize) {
  SKIP_IF_NO_CUDA_DEVICE();

  const int width = 16;
  const int height = 9;

  RenderConfig c;
  c.width = width;
  c.height = height;
  const blackhole::Image image = blackhole::render(c);

  EXPECT_EQ(image.width(), width);
  EXPECT_EQ(image.height(), height);
}

TEST(RenderTest, RenderFlatReturnsImageWithRequestedSize) {
  SKIP_IF_NO_CUDA_DEVICE();
  RenderConfig c;
  c.width = 16;
  c.height = 9;
  const blackhole::Image image = blackhole::renderFlat(c);
  EXPECT_EQ(image.width(), 16);
  EXPECT_EQ(image.height(), 9);
}
