#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "blackhole/post.hpp"
#include "cuda_test_support.hpp"

using namespace blackhole::post;

namespace {

struct Out {
  float aces[3];
  float srgb;
  float lin;
  int valid[2];
  float pipe[3];
  std::uint8_t rgba[4];
  float dx, dy;
  float bloom[3];
};

__global__ void postKernel(const float* in, Out* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float v = in[i];
  Out o;
  const Vec3 c{v, 0.5f * v, 2.0f * v};
  const Vec3 a = acesFilm(c);
  o.aces[0] = a.x;
  o.aces[1] = a.y;
  o.aces[2] = a.z;
  o.srgb = linearToSrgb(v);
  o.lin = srgbToLinear(v);
  o.valid[0] = isValidSpp(i) ? 1 : 0;
  o.valid[1] = isValidSpp(i * i - (i % 3 == 0 ? 1 : 0)) ? 1 : 0;
  const Vec3 p = postProcessPixel(c, 1.5f);
  o.pipe[0] = p.x;
  o.pipe[1] = p.y;
  o.pipe[2] = p.z;
  packRgba8(p, o.rgba);
  const SampleOffset s = sampleOffset(i % 16, 16, i, 3 * i + 1);
  o.dx = s.dx;
  o.dy = s.dy;
  const Vec3 b = bloomCombine(c, a, 0.25f);
  o.bloom[0] = b.x;
  o.bloom[1] = b.y;
  o.bloom[2] = b.z;
  out[i] = o;
}

}  // namespace

TEST(PostDeviceTest, MatchesHost) {
  SKIP_IF_NO_CUDA_DEVICE();
  const int n = 256;
  std::vector<float> in(n);
  for (int i = 0; i < n; ++i) in[i] = 0.01f * static_cast<float>(i);

  float* dIn = nullptr;
  Out* dOut = nullptr;
  ASSERT_EQ(cudaMalloc(&dIn, n * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dOut, n * sizeof(Out)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dIn, in.data(), n * sizeof(float),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  postKernel<<<(n + 63) / 64, 64>>>(dIn, dOut, n);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<Out> got(n);
  ASSERT_EQ(cudaMemcpy(got.data(), dOut, n * sizeof(Out),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  cudaFree(dIn);
  cudaFree(dOut);

  for (int i = 0; i < n; ++i) {
    const float v = in[i];
    const Vec3 c{v, 0.5f * v, 2.0f * v};
    const Vec3 a = acesFilm(c);
    const Vec3 p = postProcessPixel(c, 1.5f);
    std::uint8_t rgba[4];
    packRgba8(p, rgba);
    const SampleOffset s = sampleOffset(i % 16, 16, i, 3 * i + 1);
    const Vec3 b = bloomCombine(c, a, 0.25f);
    const Out& g = got[i];
    EXPECT_NEAR(g.aces[0], a.x, 1e-5f) << i;
    EXPECT_NEAR(g.aces[1], a.y, 1e-5f) << i;
    EXPECT_NEAR(g.aces[2], a.z, 1e-5f) << i;
    EXPECT_NEAR(g.srgb, linearToSrgb(v), 1e-5f) << i;
    EXPECT_NEAR(g.lin, srgbToLinear(v), 1e-5f) << i;
    EXPECT_EQ(g.valid[0], isValidSpp(i) ? 1 : 0) << i;
    EXPECT_EQ(g.valid[1], isValidSpp(i * i - (i % 3 == 0 ? 1 : 0)) ? 1 : 0)
        << i;
    EXPECT_NEAR(g.pipe[0], p.x, 1e-5f) << i;
    EXPECT_NEAR(g.pipe[1], p.y, 1e-5f) << i;
    EXPECT_NEAR(g.pipe[2], p.z, 1e-5f) << i;
    for (int k = 0; k < 4; ++k) {
      EXPECT_NEAR(static_cast<int>(g.rgba[k]), static_cast<int>(rgba[k]), 1)
          << i << " byte " << k;
    }
    EXPECT_EQ(g.rgba[3], 255) << i;
    EXPECT_NEAR(g.dx, s.dx, 1e-5f) << i;
    EXPECT_NEAR(g.dy, s.dy, 1e-5f) << i;
    EXPECT_NEAR(g.bloom[0], b.x, 1e-5f) << i;
    EXPECT_NEAR(g.bloom[1], b.y, 1e-5f) << i;
    EXPECT_NEAR(g.bloom[2], b.z, 1e-5f) << i;
  }
}
