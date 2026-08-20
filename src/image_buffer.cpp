#include <stdexcept>

#include "blackhole/image.hpp"

namespace blackhole {

Image::Image(int width, int height) : width_(width), height_(height) {
  if (width <= 0 || height <= 0) {
    throw std::invalid_argument("image dimensions must be positive");
  }
  pixels_.assign(static_cast<std::size_t>(width) *
                     static_cast<std::size_t>(height) * kChannels,
                 0);
}

std::size_t Image::pixelIndex(int x, int y) const {
  if (x < 0 || x >= width_ || y < 0 || y >= height_) {
    throw std::out_of_range("pixel coordinates outside image");
  }
  return (static_cast<std::size_t>(y) * static_cast<std::size_t>(width_) +
          static_cast<std::size_t>(x)) *
         kChannels;
}

void Image::setPixel(int x, int y, Pixel pixel) {
  const std::size_t i = pixelIndex(x, y);
  pixels_[i + 0] = pixel.r;
  pixels_[i + 1] = pixel.g;
  pixels_[i + 2] = pixel.b;
  pixels_[i + 3] = pixel.a;
}

Pixel Image::getPixel(int x, int y) const {
  const std::size_t i = pixelIndex(x, y);
  return {pixels_[i + 0], pixels_[i + 1], pixels_[i + 2], pixels_[i + 3]};
}

Image generateSolidColor(int width, int height, Pixel color) {
  Image image(width, height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      image.setPixel(x, y, color);
    }
  }
  return image;
}

Image generateCoordinateImage(int width, int height) {
  Image image(width, height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      image.setPixel(
          x, y,
          {static_cast<std::uint8_t>(x), static_cast<std::uint8_t>(y), 0, 255});
    }
  }
  return image;
}

}  // namespace blackhole
