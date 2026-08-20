#include "blackhole/render.hpp"

#include <stdexcept>

#include "cuda_test_support.hpp"

using blackhole::RenderConfig;

TEST(RenderConfigTest, ValidateAcceptsDefaults) {
  EXPECT_NO_THROW(blackhole::validate(RenderConfig{}));
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
