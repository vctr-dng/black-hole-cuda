#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "blackhole/image.hpp"

namespace blackhole {

struct RenderConfig {
  int width = 800;
  int height = 600;
  float observerDistance =
      20.0f; ///< Camera distance from the black hole (scene units).
  float fovDegrees = 60.0f;       ///< Full vertical field of view, in (0, 180).
  float elevationDegrees = 10.0f; ///< Camera elevation above the disk plane,
                                  ///< in [0, 90). 0 = edge-on.
};

/// Throws std::invalid_argument if any field of `config` is out of range.
void validate(const RenderConfig &config);

/// Phase-1 renderer: flat space, no bending. Pixel =
/// skyGradient(primaryRayDir). Throws std::invalid_argument on a bad config,
/// std::runtime_error on CUDA failure.
Image renderFlat(const RenderConfig &config);

/// Render the scene on the GPU. Currently a placeholder gradient.
/// Throws std::invalid_argument on a bad config, std::runtime_error on CUDA
/// failure.
Image render(const RenderConfig &config);

/// GPU test patterns: exercise launch/indexing/copy-back without any physics.
/// Every pixel is `color`.
Image renderSolidColorCuda(int width, int height, Pixel color);
/// Pixel (x, y) = (x, y, 0, 255), channels truncated to a byte.
Image renderTestPatternCuda(int width, int height);

/// Write an RGBA8 buffer to a PNG file via stb_image_write.
bool write_png(const std::string &path, int width, int height,
               const std::uint8_t *rgba);

/// Write an Image to a PNG file. Returns false on bad arguments or I/O failure.
bool write_png(const std::string &path, const Image &image);

} // namespace blackhole
