// GPU tests for the sky module: SkyTexture RAII, hardware-filtered sampling vs
// the CPU reference, and host/device parity of the BH_HD functions.
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "blackhole/sky.hpp"
#include "cuda_test_support.hpp"

namespace sky = blackhole::sky;
using sky::SkyImage;
using sky::SkyTexture;
using sky::Vec3;

namespace {

constexpr double kPi = 3.14159265358979323846;

__global__ void sampleKernel(cudaTextureObject_t tex, const Vec3* dirs,
                             Vec3* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = sky::sampleSkyDevice(tex, dirs[i]);
}

__global__ void parityKernel(const Vec3* dirs, Vec3* stars, float2* uv, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  stars[i] = sky::starfield(dirs[i]);
  float u, v;
  sky::dirToEquirect(dirs[i], u, v);
  uv[i] = make_float2(u, v);
}

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(const std::vector<T>& host) : n_(host.size()) {
    EXPECT_EQ(cudaMalloc(&ptr_, n_ * sizeof(T)), cudaSuccess);
    EXPECT_EQ(
        cudaMemcpy(ptr_, host.data(), n_ * sizeof(T), cudaMemcpyHostToDevice),
        cudaSuccess);
  }
  explicit DeviceBuffer(size_t n) : n_(n) {
    EXPECT_EQ(cudaMalloc(&ptr_, n_ * sizeof(T)), cudaSuccess);
  }
  ~DeviceBuffer() { cudaFree(ptr_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* get() const { return ptr_; }
  std::vector<T> download() const {
    std::vector<T> host(n_);
    EXPECT_EQ(
        cudaMemcpy(host.data(), ptr_, n_ * sizeof(T), cudaMemcpyDeviceToHost),
        cudaSuccess);
    return host;
  }

 private:
  T* ptr_ = nullptr;
  size_t n_;
};

std::vector<Vec3> sampleOnDevice(cudaTextureObject_t tex,
                                 const std::vector<Vec3>& dirs) {
  DeviceBuffer<Vec3> d_dirs(dirs);
  DeviceBuffer<Vec3> d_out(dirs.size());
  const int n = static_cast<int>(dirs.size());
  sampleKernel<<<(n + 127) / 128, 128>>>(tex, d_dirs.get(), d_out.get(), n);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  return d_out.download();
}

Vec3 dirFromUV(double u, double v) {
  const double lon = (u - 0.5) * 2.0 * kPi;
  const double lat = (0.5 - v) * kPi;
  return {static_cast<float>(std::cos(lat) * std::sin(lon)),
          static_cast<float>(std::sin(lat)),
          static_cast<float>(std::cos(lat) * std::cos(lon))};
}

std::vector<Vec3> randomDirs(int n, std::uint32_t seed) {
  std::mt19937 gen(seed);
  auto uni = [&] { return static_cast<double>(gen() >> 8) / 16777216.0; };
  std::vector<Vec3> dirs;
  for (int i = 0; i < n; ++i) {
    const double y = 2 * uni() - 1, phi = 2 * kPi * uni();
    const double s = std::sqrt(1 - y * y);
    dirs.push_back({static_cast<float>(s * std::sin(phi)),
                    static_cast<float>(y),
                    static_cast<float>(s * std::cos(phi))});
  }
  return dirs;
}

// Hardware bilinear filtering uses 1.8 fixed-point weights (8 fractional
// bits): each weight is off by at most 2^-9, so a blended value is off by at
// most 2^-9 * |texel difference| per axis. On makeSyntheticSky(64, 32)
// neighbouring texels differ by <= 1/31 except across the u seam (column 63 vs
// column 0: difference 1.0), giving <= ~2/512 = 0.004 there. 2/255 = 0.0078
// covers that with ~2x margin; a point-sampled or wrongly-wrapped texture is
// off by >= 1/(2*63) .. 0.5 and is caught elsewhere by exact-center and
// seam-range checks.
constexpr float kHwTol = 2.0f / 255.0f;

TEST(SkyTextureGpuTest, MatchesCpuBilinearOnRandomDirections) {
  SKIP_IF_NO_CUDA_DEVICE();
  const SkyImage img = sky::makeSyntheticSky(64, 32);
  SkyTexture tex(img);
  ASSERT_NE(tex.texture(), 0u);

  const std::vector<Vec3> dirs = randomDirs(2048, 2024);
  const std::vector<Vec3> got = sampleOnDevice(tex.texture(), dirs);
  ASSERT_EQ(got.size(), dirs.size());
  // Non-vacuity: the image is a ramp, so the samples must span most of it.
  float rMin = 1, rMax = 0, gMin = 1, gMax = 0;
  for (const Vec3& c : got) {
    rMin = std::min(rMin, c.x);
    rMax = std::max(rMax, c.x);
    gMin = std::min(gMin, c.y);
    gMax = std::max(gMax, c.y);
  }
  ASSERT_GT(rMax - rMin, 0.9f);
  ASSERT_GT(gMax - gMin, 0.9f);
  for (size_t i = 0; i < dirs.size(); ++i) {
    const Vec3 want = sky::sampleSky(img, dirs[i]);
    ASSERT_TRUE(std::isfinite(got[i].x) && std::isfinite(got[i].y) &&
                std::isfinite(got[i].z))
        << i;
    ASSERT_NEAR(got[i].x, want.x, kHwTol) << i;
    ASSERT_NEAR(got[i].y, want.y, kHwTol) << i;
    ASSERT_NEAR(got[i].z, want.z, kHwTol) << i;
  }
}

TEST(SkyTextureGpuTest, PoleAndSeamDirectionsAreFiniteAndMatchCpu) {
  SKIP_IF_NO_CUDA_DEVICE();
  const SkyImage img = sky::makeSyntheticSky(64, 32);
  SkyTexture tex(img);
  const std::vector<Vec3> dirs = {
      {0, 1, 0},           {0, -1, 0},     {0, 1.0001f, 0}, {0, -1.0001f, 0},
      {0, 0, -1},          {1e-4f, 0, -1}, {-1e-4f, 0, -1}, {1e-6f, 0.3f, -1},
      {-1e-6f, -0.3f, -1}, {0, 0, 1},      {1, 0, 0},       {-1, 0, 0}};
  const std::vector<Vec3> got = sampleOnDevice(tex.texture(), dirs);
  for (size_t i = 0; i < dirs.size(); ++i) {
    const Vec3 want = sky::sampleSky(img, dirs[i]);
    ASSERT_TRUE(std::isfinite(got[i].x) && std::isfinite(got[i].y) &&
                std::isfinite(got[i].z))
        << i;
    EXPECT_NEAR(got[i].x, want.x, kHwTol) << "dir " << i;
    EXPECT_NEAR(got[i].y, want.y, kHwTol) << "dir " << i;
    EXPECT_NEAR(got[i].z, want.z, kHwTol) << "dir " << i;
  }
  // Pole rows: g (= y/(h-1)) is 0 at the zenith row and 1 at the nadir row.
  EXPECT_NEAR(got[0].y, 0.0f, kHwTol);
  EXPECT_NEAR(got[1].y, 1.0f, kHwTol);
}

TEST(SkyTextureGpuTest, WrapsInUAndClampsInVOnHardware) {
  SKIP_IF_NO_CUDA_DEVICE();
  // r = x/63 runs 0..1 across the width, so wrap-vs-clamp is unmistakable.
  const SkyImage img = sky::makeSyntheticSky(64, 32);
  SkyTexture tex(img);
  // u = 1 - 1.6e-4 and 1.6e-4 straddle the seam: both sit ~0.5 texel from each
  // of column 63 (r = 1) and column 0 (r = 0), so r ~ 0.5, not 0 or 1.
  const std::vector<Vec3> dirs = {{1e-3f, 0, -1}, {-1e-3f, 0, -1}};
  const std::vector<Vec3> got = sampleOnDevice(tex.texture(), dirs);
  EXPECT_GT(got[0].x, 0.3f);
  EXPECT_LT(got[0].x, 0.7f);
  EXPECT_GT(got[1].x, 0.3f);
  EXPECT_LT(got[1].x, 0.7f);
  // Clamp in v: just below the zenith / above the nadir, g is the edge row.
  const std::vector<Vec3> poles = {dirFromUV(0.3, 0.001),
                                   dirFromUV(0.3, 0.999)};
  const std::vector<Vec3> p = sampleOnDevice(tex.texture(), poles);
  EXPECT_NEAR(p[0].y, 0.0f, kHwTol);
  EXPECT_NEAR(p[1].y, 1.0f, kHwTol);
}

TEST(SkyTextureGpuTest, TexelCentersAreExactAndBetweenCentersBlend) {
  SKIP_IF_NO_CUDA_DEVICE();
  const SkyImage img = sky::makeSyntheticSky(64, 32);
  SkyTexture tex(img);
  std::vector<Vec3> dirs;
  std::vector<std::pair<int, int>> texels = {{0, 0},   {63, 31}, {10, 7},
                                             {40, 20}, {0, 31},  {63, 0}};
  for (auto [x, y] : texels)
    dirs.push_back(dirFromUV((x + 0.5) / 64, (y + 0.5) / 32));
  const std::vector<Vec3> got = sampleOnDevice(tex.texture(), dirs);
  for (size_t i = 0; i < texels.size(); ++i) {
    const auto [x, y] = texels[i];
    // At a texel center the hardware weight is exactly 0 -> exact texel (up to
    // float rounding of the direction -> u,v conversion, ~1e-5 here).
    EXPECT_NEAR(got[i].x, x / 63.0f, 5e-4f) << x << "," << y;
    EXPECT_NEAR(got[i].y, y / 31.0f, 5e-4f) << x << "," << y;
    EXPECT_NEAR(got[i].z, 0.25f, 1e-6f);
  }
  // Halfway between two texel centers in u: r = (x + 0.5)/63.
  const std::vector<Vec3> mid =
      sampleOnDevice(tex.texture(), {dirFromUV(11.0 / 64, (7 + 0.5) / 32)});
  EXPECT_NEAR(mid[0].x, 10.5f / 63.0f, kHwTol);
}

TEST(SkyTextureGpuTest, NonSquareImageMatchesCpu) {
  SKIP_IF_NO_CUDA_DEVICE();
  // Odd, non-square, non-power-of-two size: catches width/height or pitch
  // mix-ups in the upload. Same hardware-weight tolerance argument as above
  // (texel steps are even smaller than on 64x32 except across the seam).
  const SkyImage img = sky::makeSyntheticSky(10, 3);
  SkyTexture tex(img);
  const std::vector<Vec3> dirs = randomDirs(1024, 99);
  const std::vector<Vec3> got = sampleOnDevice(tex.texture(), dirs);
  float gMax = 0;
  for (size_t i = 0; i < dirs.size(); ++i) {
    const Vec3 want = sky::sampleSky(img, dirs[i]);
    ASSERT_NEAR(got[i].x, want.x, kHwTol) << i;
    ASSERT_NEAR(got[i].y, want.y, kHwTol) << i;
    ASSERT_NEAR(got[i].z, want.z, kHwTol) << i;
    gMax = std::max(gMax, got[i].y);
  }
  EXPECT_GT(gMax, 0.8f);  // non-vacuous: rows really vary
}

TEST(SkyTextureGpuTest, ConstructorRejectsInvalidImages) {
  SKIP_IF_NO_CUDA_DEVICE();
  EXPECT_THROW(SkyTexture{SkyImage{}}, std::invalid_argument);

  SkyImage zeroW = sky::makeSyntheticSky(4, 4);
  zeroW.width = 0;
  EXPECT_THROW(SkyTexture{zeroW}, std::invalid_argument);

  SkyImage negH = sky::makeSyntheticSky(4, 4);
  negH.height = -4;
  EXPECT_THROW(SkyTexture{negH}, std::invalid_argument);

  SkyImage tooSmall = sky::makeSyntheticSky(4, 4);
  tooSmall.rgb.pop_back();  // size != w*h*3
  EXPECT_THROW(SkyTexture{tooSmall}, std::invalid_argument);

  SkyImage tooBig = sky::makeSyntheticSky(4, 4);
  tooBig.rgb.push_back(0.0f);
  EXPECT_THROW(SkyTexture{tooBig}, std::invalid_argument);

  SkyImage noPixels = sky::makeSyntheticSky(4, 4);
  noPixels.rgb.clear();  // dims claim 4x4 but no data
  EXPECT_THROW(SkyTexture{noPixels}, std::invalid_argument);

  // A valid image still works afterwards (no sticky error / leaked state).
  EXPECT_NO_THROW(SkyTexture{sky::makeSyntheticSky(4, 4)});
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST(SkyTextureGpuTest, IsMoveOnly) {
  static_assert(!std::is_copy_constructible_v<SkyTexture>);
  static_assert(!std::is_copy_assignable_v<SkyTexture>);
  static_assert(std::is_nothrow_move_constructible_v<SkyTexture>);
  static_assert(std::is_nothrow_move_assignable_v<SkyTexture>);
  SUCCEED();
}

TEST(SkyTextureGpuTest, MoveConstructionTransfersOwnership) {
  SKIP_IF_NO_CUDA_DEVICE();
  const SkyImage img = sky::makeSyntheticSky(16, 8);
  SkyTexture a(img);
  const cudaTextureObject_t handle = a.texture();
  ASSERT_NE(handle, 0u);

  SkyTexture b(std::move(a));
  EXPECT_EQ(a.texture(), 0u);  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(b.texture(), handle);

  // The texture is still alive and usable through the new owner.
  const std::vector<Vec3> got =
      sampleOnDevice(b.texture(), {dirFromUV(0.5 / 16, 0.5 / 8)});
  EXPECT_NEAR(got[0].x, 0.0f, 1e-3f);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST(SkyTextureGpuTest, MoveAssignmentReleasesOldAndTransfers) {
  SKIP_IF_NO_CUDA_DEVICE();
  SkyTexture a(sky::makeSyntheticSky(16, 8));
  SkyTexture b(sky::makeSyntheticSky(32, 16));
  const cudaTextureObject_t hb = b.texture();
  ASSERT_NE(hb, 0u);
  ASSERT_NE(a.texture(), hb);

  a = std::move(b);  // old texture of `a` must be freed, not leaked/aliased
  EXPECT_EQ(a.texture(), hb);
  EXPECT_EQ(b.texture(), 0u);  // NOLINT(bugprone-use-after-move)

  // Moved-into object is usable; destruction of both (incl. the moved-from)
  // must not double free: every CUDA call afterwards stays successful.
  const std::vector<Vec3> got =
      sampleOnDevice(a.texture(), {dirFromUV(0.5 / 32, 0.5 / 16)});
  EXPECT_NEAR(got[0].x, 0.0f, 1e-3f);
  {
    SkyTexture sink(std::move(a));
  }
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  void* p = nullptr;
  EXPECT_EQ(cudaMalloc(&p, 16), cudaSuccess);
  cudaFree(p);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

TEST(SkyDeviceParityTest, HostDeviceFunctionsAgree) {
  SKIP_IF_NO_CUDA_DEVICE();
  const int n = 4096;
  const std::vector<Vec3> dirs = randomDirs(n, 4242);
  DeviceBuffer<Vec3> d_dirs(dirs);
  DeviceBuffer<Vec3> d_stars(n);
  DeviceBuffer<float2> d_uv(n);
  parityKernel<<<(n + 127) / 128, 128>>>(d_dirs.get(), d_stars.get(),
                                         d_uv.get(), n);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  const auto stars = d_stars.download();
  const auto uv = d_uv.download();

  int starMismatch = 0;
  for (int i = 0; i < n; ++i) {
    float u, v;
    sky::dirToEquirect(dirs[i], u, v);
    // Device and host libm differ by a few ulp (~1e-7); 1e-4 is generous.
    // u near the seam can flip between ~0 and ~1: compare circularly.
    const float du = std::fabs(u - uv[i].x);
    EXPECT_TRUE(du < 1e-4f || std::fabs(du - 1.0f) < 1e-4f) << i;
    EXPECT_NEAR(v, uv[i].y, 1e-4f) << i;

    const Vec3 h = sky::starfield(dirs[i]);
    ASSERT_TRUE(std::isfinite(stars[i].x) && std::isfinite(stars[i].y) &&
                std::isfinite(stars[i].z) && stars[i].x >= 0 &&
                stars[i].y >= 0 && stars[i].z >= 0)
        << i;
    // FMA contraction on the device can shift a star blob's edge by ~1e-6
    // relative, so allow a tiny fraction of samples (<1%) to disagree.
    if (std::fabs(h.x - stars[i].x) > 1e-2f ||
        std::fabs(h.y - stars[i].y) > 1e-2f ||
        std::fabs(h.z - stars[i].z) > 1e-2f) {
      ++starMismatch;
    }
  }
  EXPECT_LT(starMismatch, n / 100);
}

}  // namespace
