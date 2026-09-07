#include "blackhole/sky.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace blackhole::sky {
namespace {

void check(cudaError_t err, const char *what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(err));
  }
}

float srgbToLinear(float c) {
  return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

} // namespace

SkyImage loadSkyImage(const std::string &path) {
  int w = 0, h = 0, n = 0;
  SkyImage img;
  if (stbi_is_hdr(path.c_str())) {
    float *data = stbi_loadf(path.c_str(), &w, &h, &n, 3);
    if (!data) {
      throw std::runtime_error("failed to load sky image '" + path +
                               "': " + stbi_failure_reason());
    }
    img.rgb.assign(data, data + static_cast<size_t>(w) * h * 3);
    stbi_image_free(data);
  } else {
    unsigned char *data = stbi_load(path.c_str(), &w, &h, &n, 3);
    if (!data) {
      throw std::runtime_error("failed to load sky image '" + path +
                               "': " + stbi_failure_reason());
    }
    img.rgb.resize(static_cast<size_t>(w) * h * 3);
    for (size_t i = 0; i < img.rgb.size(); ++i) {
      img.rgb[i] = srgbToLinear(static_cast<float>(data[i]) / 255.0f);
    }
    stbi_image_free(data);
  }
  img.width = w;
  img.height = h;
  return img;
}

SkyImage makeSyntheticSky(int w, int h) {
  SkyImage img;
  img.width = w;
  img.height = h;
  img.rgb.resize(static_cast<size_t>(w) * h * 3);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      float *p = &img.rgb[(static_cast<size_t>(y) * w + x) * 3];
      p[0] = static_cast<float>(x) / static_cast<float>(w - 1);
      p[1] = static_cast<float>(y) / static_cast<float>(h - 1);
      p[2] = 0.25f;
    }
  }
  return img;
}

Vec3 sampleSky(const SkyImage &image, Vec3 dir) {
  float u, v;
  dirToEquirect(dir, u, v);
  const float fx = u * static_cast<float>(image.width) - 0.5f;
  const float fy = v * static_cast<float>(image.height) - 0.5f;
  const float x0f = std::floor(fx), y0f = std::floor(fy);
  const float wx = fx - x0f, wy = fy - y0f;
  auto wrapx = [&](int x) {
    return ((x % image.width) + image.width) % image.width;
  };
  auto clampy = [&](int y) {
    return y < 0 ? 0 : (y >= image.height ? image.height - 1 : y);
  };
  const int xs[2] = {wrapx(static_cast<int>(x0f)),
                     wrapx(static_cast<int>(x0f) + 1)};
  const int ys[2] = {clampy(static_cast<int>(y0f)),
                     clampy(static_cast<int>(y0f) + 1)};
  float out[3] = {0, 0, 0};
  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < 2; ++i) {
      const float wgt = (i ? wx : 1 - wx) * (j ? wy : 1 - wy);
      const float *p =
          &image.rgb[(static_cast<size_t>(ys[j]) * image.width + xs[i]) * 3];
      for (int c = 0; c < 3; ++c)
        out[c] += wgt * p[c];
    }
  }
  return {out[0], out[1], out[2]};
}

SkyTexture::SkyTexture(const SkyImage &image) {
  if (image.width <= 0 || image.height <= 0 ||
      image.rgb.size() != static_cast<size_t>(image.width) * image.height * 3) {
    throw std::invalid_argument("SkyTexture: empty or ill-sized image");
  }
  std::vector<float> rgba(static_cast<size_t>(image.width) * image.height * 4);
  for (size_t i = 0; i < static_cast<size_t>(image.width) * image.height; ++i) {
    rgba[i * 4 + 0] = image.rgb[i * 3 + 0];
    rgba[i * 4 + 1] = image.rgb[i * 3 + 1];
    rgba[i * 4 + 2] = image.rgb[i * 3 + 2];
    rgba[i * 4 + 3] = 1.0f;
  }
  const cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
  check(cudaMallocArray(&array_, &desc, image.width, image.height),
        "cudaMallocArray");
  try {
    const size_t pitch = static_cast<size_t>(image.width) * sizeof(float4);
    check(cudaMemcpy2DToArray(array_, 0, 0, rgba.data(), pitch, pitch,
                              image.height, cudaMemcpyHostToDevice),
          "cudaMemcpy2DToArray");
    cudaResourceDesc res{};
    res.resType = cudaResourceTypeArray;
    res.res.array.array = array_;
    cudaTextureDesc tex{};
    tex.addressMode[0] = cudaAddressModeWrap;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.filterMode = cudaFilterModeLinear;
    tex.readMode = cudaReadModeElementType;
    tex.normalizedCoords = 1;
    check(cudaCreateTextureObject(&texture_, &res, &tex, nullptr),
          "cudaCreateTextureObject");
  } catch (...) {
    release();
    throw;
  }
}

SkyTexture::~SkyTexture() { release(); }

SkyTexture::SkyTexture(SkyTexture &&o) noexcept
    : array_(std::exchange(o.array_, nullptr)),
      texture_(std::exchange(o.texture_, 0)) {}

SkyTexture &SkyTexture::operator=(SkyTexture &&o) noexcept {
  if (this != &o) {
    release();
    array_ = std::exchange(o.array_, nullptr);
    texture_ = std::exchange(o.texture_, 0);
  }
  return *this;
}

void SkyTexture::release() noexcept {
  if (texture_)
    cudaDestroyTextureObject(texture_);
  if (array_)
    cudaFreeArray(array_);
  texture_ = 0;
  array_ = nullptr;
}

} // namespace blackhole::sky
