#include <gtest/gtest.h>

#include "blackhole/image.hpp"

using blackhole::generateCoordinateImage;
using blackhole::generateSolidColor;
using blackhole::Image;
using blackhole::Pixel;

TEST(ImageGeneratorTest, GeneratesSolidRed) {
  const int width = 16;
  const int height = 16;
  const Image image = generateSolidColor(width, height, {255, 0, 0, 255});

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

TEST(ImageGeneratorTest, EncodesCoordinates) {
  const int width = 16;
  const int height = 16;
  const Image image = generateCoordinateImage(width, height);

  const Pixel p = image.getPixel(8, 8);
  EXPECT_EQ(p.r, 8);
  EXPECT_EQ(p.g, 8);
  EXPECT_EQ(p.b, 0);
  EXPECT_EQ(p.a, 255);
}

TEST(ImageGeneratorTest, CoordinateImageIsNotTransposed) {
  // Non-square so that a row/column swap cannot go unnoticed.
  const int width = 16;
  const int height = 9;
  const Image image = generateCoordinateImage(width, height);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const Pixel p = image.getPixel(x, y);
      EXPECT_EQ(p.r, x);
      EXPECT_EQ(p.g, y);
    }
  }
}
