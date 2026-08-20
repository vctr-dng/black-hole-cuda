#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace blackhole {

struct Pixel {
  std::uint8_t r = 0;
  std::uint8_t g = 0;
  std::uint8_t b = 0;
  std::uint8_t a = 255;
};

/// Row-major RGBA8 image.
/// Pixel (x, y) starts at byte (y * width + x) * 4.
class Image {
 public:
  static constexpr int kChannels = 4;

  Image(int width, int height);

  int width() const { return width_; }
  int height() const { return height_; }
  std::size_t size() const { return pixels_.size(); }

  /// Byte offset of pixel (x, y)
  std::size_t pixelIndex(int x, int y) const;

  void setPixel(int x, int y, Pixel pixel);
  Pixel getPixel(int x, int y) const;

  std::uint8_t *data() { return pixels_.data(); }
  const std::uint8_t *data() const { return pixels_.data(); }

 private:
  int width_;
  int height_;
  std::vector<std::uint8_t> pixels_;
};

Image generateSolidColor(int width, int height, Pixel color);

/// Image where pixel (x, y) = (x, y, 0, 255), each channel truncated to a byte.
Image generateCoordinateImage(int width, int height);

}  // namespace blackhole
