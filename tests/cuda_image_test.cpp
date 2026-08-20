#include <utility>

#include "blackhole/render.hpp"
#include "cuda_test_support.hpp"

using blackhole::Image;
using blackhole::Pixel;

TEST(CudaImageTest, SolidColorKernelFillsEveryPixel) {
  SKIP_IF_NO_CUDA_DEVICE();

  const int width = 8;
  const int height = 8;
  const Image image =
      blackhole::renderSolidColorCuda(width, height, {255, 0, 0, 255});

  ASSERT_EQ(image.width(), width);
  ASSERT_EQ(image.height(), height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const Pixel p = image.getPixel(x, y);
      EXPECT_EQ(p.r, 255) << x << "," << y;
      EXPECT_EQ(p.g, 0) << x << "," << y;
      EXPECT_EQ(p.b, 0) << x << "," << y;
      EXPECT_EQ(p.a, 255) << x << "," << y;
    }
  }
}

TEST(CudaImageTest, CoordinateKernelEncodesXY) {
  SKIP_IF_NO_CUDA_DEVICE();

  const int width = 8;
  const int height = 8;
  const Image image = blackhole::renderTestPatternCuda(width, height);

  const Pixel p = image.getPixel(width - 1, height - 1);
  EXPECT_EQ(p.r, width - 1);
  EXPECT_EQ(p.g, height - 1);
  EXPECT_EQ(p.b, 0);
  EXPECT_EQ(p.a, 255);
}

TEST(CudaImageTest, MatchesCpuReferenceForNonBlockMultipleSizes) {
  SKIP_IF_NO_CUDA_DEVICE();

  // Block is 16x16: cover 1x1, just-over-a-block, narrow, and wide/odd sizes.
  const std::pair<int, int> sizes[] = {{1, 1}, {16, 16}, {16, 9}, {9, 16}};
  for (const auto& [w, h] : sizes) {
    SCOPED_TRACE(std::to_string(w) + "x" + std::to_string(h));

    const Image gpu = blackhole::renderTestPatternCuda(w, h);
    const Image cpu = blackhole::generateCoordinateImage(w, h);

    ASSERT_EQ(gpu.width(), w);
    ASSERT_EQ(gpu.height(), h);
    ASSERT_EQ(gpu.size(), cpu.size());
    EXPECT_TRUE(std::equal(gpu.data(), gpu.data() + gpu.size(), cpu.data()));
  }
}

TEST(CudaImageTest, LastPixelIsWrittenAndNothingIsLeftUntouched) {
  SKIP_IF_NO_CUDA_DEVICE();

  // Alpha is 255 for a pixel if the pixel was
  // - written by some thread
  // - given a memory location already set to 255 by chance
  const int width = 16;
  const int height = 9;
  const Image image = blackhole::renderTestPatternCuda(width, height);
  const Pixel last = image.getPixel(width - 1, height - 1);
  EXPECT_EQ(last.r, width - 1);
  EXPECT_EQ(last.g, height - 1);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      EXPECT_EQ(image.getPixel(x, y).a, 255) << x << "," << y;
}

TEST(CudaImageTest, RejectsInvalidDimensions) {
  EXPECT_THROW(blackhole::renderTestPatternCuda(0, 8), std::invalid_argument);
  EXPECT_THROW(blackhole::renderTestPatternCuda(8, 0), std::invalid_argument);
  EXPECT_THROW(blackhole::renderSolidColorCuda(8, -1, {}),
               std::invalid_argument);
  EXPECT_THROW(blackhole::renderSolidColorCuda(-1, 8, {}),
               std::invalid_argument);
}
