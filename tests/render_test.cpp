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
