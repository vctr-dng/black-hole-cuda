#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

#include "blackhole/camera.hpp"
#include "blackhole/disk.hpp"
#include "blackhole/geodesic.hpp"
#include "blackhole/render.hpp"
#include "blackhole/sky.hpp"

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

BH_HD geo::Vec3 to_geo(cam::Vec3 v) { return {v.x, v.y, v.z}; }
BH_HD cam::Vec3 to_cam(geo::Vec3 v) { return {v.x, v.y, v.z}; }

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

__device__ std::uint8_t to_byte_clamped(float c) {
  c = c < 0.0f ? 0.0f : (c > 1.0f ? 1.0f : c);
  return static_cast<std::uint8_t>(lroundf(255.0f * c));
}

BH_HD sky::Vec3 to_sky(geo::Vec3 v) { return {v.x, v.y, v.z}; }
BH_HD cam::Vec3 from_sky(sky::Vec3 v) { return {v.x, v.y, v.z}; }

/// Background colour (linear RGB) for an escaped direction `d` (unit).
__device__ cam::Vec3 background(SkyMode mode, cudaTextureObject_t tex,
                                geo::Vec3 d) {
  switch (mode) {
  case SkyMode::Starfield:
    return from_sky(sky::starfield(to_sky(d)));
  case SkyMode::Image:
    return from_sky(sky::sampleSkyDevice(tex, to_sky(d)));
  case SkyMode::Gradient:
  default:
    return cam::skyGradient(to_cam(d));
  }
}

/// Shades one primary ray: trace, then colour by outcome. Future features
/// (disk hits, supersampling) hook in here.
__device__ cam::Vec3 shade_ray(const geo::Ray &ray,
                               const geo::TraceParams &params, SkyMode mode,
                               cudaTextureObject_t tex) {
  const geo::TraceResult res = geo::trace(ray, params);
  if (res.outcome == geo::Outcome::Escaped) {
    return background(mode, tex, geo::normalize(res.ray.vel));
  }
  return {0.0f, 0.0f, 0.0f};
}

BH_HD disk::Vec3 to_disk(geo::Vec3 v) { return {v.x, v.y, v.z}; }

/// Opaque-disk variant of shade_ray: the first plane crossing inside the
/// disk's radial range terminates the ray with the disk radiance.
__device__ cam::Vec3 shade_ray_disk(geo::Ray ray,
                                    const geo::TraceParams &params,
                                    const disk::DiskParams &dp, SkyMode mode,
                                    cudaTextureObject_t tex) {
  for (int i = 0; i < params.maxSteps; ++i) {
    const geo::Vec3 p0 = ray.pos;
    geo::stepVerlet(ray, geo::adaptiveStep(geo::length(ray.pos), true));
    const disk::Crossing c = disk::planeCrossing(to_disk(p0), to_disk(ray.pos));
    if (c.hit) {
      const float r = disk::length(c.pos);
      if (disk::inDisk(r, dp)) {
        const geo::Vec3 v = geo::normalize(ray.vel);
        const float cosT = disk::dot(disk::orbitalDirection(c.pos),
                                     disk::Vec3{-v.x, -v.y, -v.z});
        const disk::Vec3 col = disk::diskRadiance(r, cosT, dp);
        return {col.x, col.y, col.z};
      }
    }
    const float rad = geo::length(ray.pos);
    if (rad <= geo::kRs) {
      return {0.0f, 0.0f, 0.0f};
    }
    if (rad >= params.rEscape && geo::dot(ray.pos, ray.vel) > 0.0f) {
      return background(mode, tex, geo::normalize(ray.vel));
    }
  }
  return {0.0f, 0.0f, 0.0f};
}

__global__ void lensing_kernel(std::uint8_t *rgba, int width, int height,
                               cam::Camera camera, geo::TraceParams params,
                               SkyMode mode, cudaTextureObject_t tex,
                               bool useDisk, disk::DiskParams dp) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }

  const geo::Ray ray = geo::makeRay(to_geo(camera.pos),
                                    to_geo(cam::primaryRayDir(camera, x, y)));
  const cam::Vec3 c = useDisk ? shade_ray_disk(ray, params, dp, mode, tex)
                              : shade_ray(ray, params, mode, tex);
  const int idx = (y * width + x) * 4;
  rgba[idx + 0] = to_byte_clamped(c.x);
  rgba[idx + 1] = to_byte_clamped(c.y);
  rgba[idx + 2] = to_byte_clamped(c.z);
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
  if (config.sky == SkyMode::Image && config.skyImagePath.empty()) {
    throw std::invalid_argument(
        "skyImagePath must be non-empty when sky mode is image");
  }
  if (!(std::isfinite(config.diskInner) && config.diskInner > 1.0f)) {
    throw std::invalid_argument("diskInner must be finite and > 1");
  }
  if (!(std::isfinite(config.diskOuter) &&
        config.diskOuter > config.diskInner)) {
    throw std::invalid_argument("diskOuter must be finite and > diskInner");
  }
  if (!(config.diskTempK >= 1000.0f && config.diskTempK <= 40000.0f)) {
    throw std::invalid_argument("diskTempK must be in [1000, 40000] Kelvin");
  }
  if (!(std::isfinite(config.diskBrightness) &&
        config.diskBrightness >= 0.0f)) {
    throw std::invalid_argument("diskBrightness must be finite and >= 0");
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

geo::TraceParams renderTraceParams(const RenderConfig &config) {
  geo::TraceParams p;
  p.rEscape = std::max(40.0f, 2.0f * config.observerDistance);
  p.maxSteps = 4000;
  p.nearDisk = false;
  return p;
}

Image render(const RenderConfig &config) {
  validate(config);
  const cam::Camera camera =
      cam::makeCamera(config.width, config.height, config.observerDistance,
                      config.fovDegrees, config.elevationDegrees);
  const geo::TraceParams params = renderTraceParams(config);
  // Load before any GPU work so a bad path throws early; RAII frees it.
  std::unique_ptr<sky::SkyTexture> texture;
  if (config.sky == SkyMode::Image) {
    texture = std::make_unique<sky::SkyTexture>(
        sky::loadSkyImage(config.skyImagePath));
  }
  const cudaTextureObject_t tex = texture ? texture->texture() : 0;
  const disk::DiskParams dp{config.diskInner, config.diskOuter,
                            config.diskTempK, config.diskBrightness};
  return run_on_device(config.width, config.height, "lensing_kernel",
                       [&](std::uint8_t *d, dim3 grid, dim3 block) {
                         lensing_kernel<<<grid, block>>>(
                             d, config.width, config.height, camera, params,
                             config.sky, tex, config.disk, dp);
                       });
}

} // namespace blackhole
