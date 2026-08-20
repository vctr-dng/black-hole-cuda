#include "blackhole/render.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace blackhole {

bool write_png(const std::string& path, int width, int height,
               const std::uint8_t* rgba) {
  if (width <= 0 || height <= 0 || rgba == nullptr || path.empty()) {
    return false;
  }

  const int stride = width * 4;
  return stbi_write_png(path.c_str(), width, height, 4, rgba, stride) != 0;
}

bool write_png(const std::string& path, const Image& image) {
  return write_png(path, image.width(), image.height(), image.data());
}

}  // namespace blackhole
