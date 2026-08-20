#include "blackhole/image.hpp"

#include <gtest/gtest.h>

#include <stdexcept>

using blackhole::Image;
using blackhole::Pixel;

TEST(ImageTest, AllocatesCorrectBufferSize) {
  const int width = 8;
  const int height = 8;

  Image image(width, height);

  EXPECT_EQ(image.width(), width);
  EXPECT_EQ(image.height(), height);
  EXPECT_EQ(image.size(), static_cast<std::size_t>(width) *
                              static_cast<std::size_t>(height) * 4u);
}

TEST(ImageTest, RejectsInvalidDimensions) {
  EXPECT_THROW(Image(0, 100), std::invalid_argument);
  EXPECT_THROW(Image(100, 0), std::invalid_argument);
  EXPECT_THROW(Image(-1, 100), std::invalid_argument);
  EXPECT_THROW(Image(100, -1), std::invalid_argument);
}

TEST(ImageTest, PixelIndexing) {
  const int width = 8;
  const int height = 8;

  Image image(width, height);

  EXPECT_EQ(image.pixelIndex(0, 0), 0u);
  EXPECT_EQ(image.pixelIndex(1, 0), 4u);
  EXPECT_EQ(image.pixelIndex(0, 1), static_cast<std::size_t>(width) * 4u);
  EXPECT_EQ(
      image.pixelIndex(height - 1, width - 1),
      (static_cast<std::size_t>(width) * static_cast<std::size_t>(height - 1) +
       static_cast<std::size_t>(width - 1)) *
          4u);
  EXPECT_EQ(image.pixelIndex(height - 1, width - 1) + 4u, image.size());
}

TEST(ImageTest, PixelIndexRejectsOutOfRange) {
  const int width = 8;
  const int height = 8;
  Image image(width, height);

  EXPECT_THROW(image.pixelIndex(width, 0), std::out_of_range);
  EXPECT_THROW(image.pixelIndex(0, height), std::out_of_range);
  EXPECT_THROW(image.pixelIndex(-1, 0), std::out_of_range);
  EXPECT_THROW(image.pixelIndex(0, -1), std::out_of_range);
}

TEST(ImageTest, SetsAndGetsPixel) {
  const int width = 8;
  const int height = 8;
  Image image(width, height);

  image.setPixel(1, 0, {255, 128, 64, 32});

  const Pixel p = image.getPixel(1, 0);
  EXPECT_EQ(p.r, 255);
  EXPECT_EQ(p.g, 128);
  EXPECT_EQ(p.b, 64);
  EXPECT_EQ(p.a, 32);
}

TEST(ImageTest, SetPixelDoesNotTouchOtherPixels) {
  const int width = 8;
  const int height = 8;
  Image image(width, height);

  image.setPixel(1, 0, {255, 255, 255, 128});

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (x == 1 && y == 0) continue;
      const Pixel p = image.getPixel(x, y);
      EXPECT_EQ(p.r, 0) << x << "," << y;
      EXPECT_EQ(p.g, 0) << x << "," << y;
      EXPECT_EQ(p.b, 0) << x << "," << y;
      EXPECT_EQ(p.a, 0) << x << "," << y;
    }
  }
}

TEST(ImageTest, ExposesRawBuffer) {
  const int width = 8;
  const int height = 8;
  Image image(width, height);
  image.setPixel(1, 1, {1, 2, 3, 4});

  const std::size_t i = image.pixelIndex(1, 1);
  EXPECT_EQ(image.data()[i + 0], 1);
  EXPECT_EQ(image.data()[i + 1], 2);
  EXPECT_EQ(image.data()[i + 2], 3);
  EXPECT_EQ(image.data()[i + 3], 4);
}
