#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "blackhole/camera.hpp"
#include "blackhole/render.hpp"

namespace blackhole {
namespace {

void check_cuda(cudaError_t err, const char *what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(err));
  }
}

__device__ std::uint8_t to_byte(float c) {
  return static_cast<std::uint8_t>(lroundf(255.0f * c));
}

__global__ void flat_sky_kernel(std::uint8_t *rgba, int width, int height,
                                cam::Camera camera) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }

  const cam::Vec3 c = cam::skyGradient(cam::primaryRayDir(camera, x, y));
  const int idx = (y * width + x) * 4;
  rgba[idx + 0] = to_byte(c.x);
  rgba[idx + 1] = to_byte(c.y);
  rgba[idx + 2] = to_byte(c.z);
  rgba[idx + 3] = 255;
}

__global__ void fill_placeholder_kernel(std::uint8_t *rgba, int width,
                                        int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }

  const int idx = (y * width + x) * 4;
  const float u = static_cast<float>(x) / static_cast<float>(width);
  const float v = static_cast<float>(y) / static_cast<float>(height);

  rgba[idx + 0] = static_cast<std::uint8_t>(u * 255.0f);
  rgba[idx + 1] = static_cast<std::uint8_t>(v * 255.0f);
  rgba[idx + 2] = 40;
  rgba[idx + 3] = 255;
}

__global__ void solid_color_kernel(std::uint8_t *rgba, int width, int height,
                                   Pixel color) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }

  const int idx = (y * width + x) * 4;
  rgba[idx + 0] = color.r;
  rgba[idx + 1] = color.g;
  rgba[idx + 2] = color.b;
  rgba[idx + 3] = color.a;
}

__global__ void coordinate_kernel(std::uint8_t *rgba, int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }

  const int idx = (y * width + x) * 4;
  rgba[idx + 0] = static_cast<std::uint8_t>(x);
  rgba[idx + 1] = static_cast<std::uint8_t>(y);
  rgba[idx + 2] = 0;
  rgba[idx + 3] = 255;
}

// Runs `launch(device_rgba, grid, block)` over a fresh device buffer and copies
// it back.
template <typename Launch>
Image run_on_device(int width, int height, const char *name, Launch launch) {
  Image image(width, height); // validates dimensions

  std::uint8_t *device_rgba = nullptr;
  check_cuda(cudaMalloc(&device_rgba, image.size()), "cudaMalloc");

  const dim3 block(16, 16);
  const dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);
  launch(device_rgba, grid, block);
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess)
    err = cudaDeviceSynchronize();
  if (err == cudaSuccess) {
    err = cudaMemcpy(image.data(), device_rgba, image.size(),
                     cudaMemcpyDeviceToHost);
  }
  cudaFree(device_rgba);
  check_cuda(err, name);
  return image;
}

} // namespace

Image renderSolidColorCuda(int width, int height, Pixel color) {
  return run_on_device(width, height, "solid_color_kernel",
                       [&](std::uint8_t *d, dim3 grid, dim3 block) {
                         solid_color_kernel<<<grid, block>>>(d, width, height,
                                                             color);
                       });
}

Image renderTestPatternCuda(int width, int height) {
  return run_on_device(width, height, "coordinate_kernel",
                       [&](std::uint8_t *d, dim3 grid, dim3 block) {
                         coordinate_kernel<<<grid, block>>>(d, width, height);
                       });
}

void validate(const RenderConfig &config) {
  if (config.width <= 0 || config.height <= 0) {
    throw std::invalid_argument("render dimensions must be positive");
  }
  if (!(config.observerDistance > 0.0f)) {
    throw std::invalid_argument("observer distance must be positive");
  }
  if (!(config.fovDegrees > 0.0f && config.fovDegrees < 180.0f)) {
    throw std::invalid_argument("field of view must be in (0, 180) degrees");
  }
  if (!(config.elevationDegrees >= 0.0f && config.elevationDegrees < 90.0f)) {
    throw std::invalid_argument("elevation must be in [0, 90) degrees");
  }
}

Image renderFlat(const RenderConfig &config) {
  validate(config);
  const cam::Camera camera =
      cam::makeCamera(config.width, config.height, config.observerDistance,
                      config.fovDegrees, config.elevationDegrees);
  return run_on_device(config.width, config.height, "flat_sky_kernel",
                       [&](std::uint8_t *d, dim3 grid, dim3 block) {
                         flat_sky_kernel<<<grid, block>>>(
                             d, config.width, config.height, camera);
                       });
}

Image render(const RenderConfig &config) {
  validate(config);
  return run_on_device(config.width, config.height, "fill_placeholder_kernel",
                       [&](std::uint8_t *d, dim3 grid, dim3 block) {
                         fill_placeholder_kernel<<<grid, block>>>(
                             d, config.width, config.height);
                       });
}

} // namespace blackhole
