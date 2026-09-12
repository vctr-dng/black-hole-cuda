// CUDA compile-smoke for include/blackhole/disk.hpp: the whole API runs inside
// a kernel and must agree with the host result.

#include <cmath>
#include <vector>

#include "blackhole/disk.hpp"
#include "cuda_test_support.hpp"

namespace disk = blackhole::disk;

namespace {

constexpr int kCases = 8;
constexpr int kFloatsPerCase = 40;

BH_HD inline void put(float* o, int& k, disk::Vec3 v) {
  o[k++] = v.x;
  o[k++] = v.y;
  o[k++] = v.z;
}

// Exercises every public function for case `i`; writes <= kFloatsPerCase.
BH_HD inline void evalCase(int i, float* o) {
  int k = 0;
  const disk::DiskParams p;
  const float r = 3.0f + 1.25f * static_cast<float>(i);
  const float c = -1.0f + 2.0f * static_cast<float>(i) / (kCases - 1);
  const disk::Vec3 a{1.0f + i, 2.0f, -3.0f};
  const disk::Vec3 b{0.5f, -1.5f * i, 2.0f};
  put(o, k, a + b);
  put(o, k, a - b);
  put(o, k, a * 2.0f);
  o[k++] = disk::dot(a, b);
  put(o, k, disk::cross(a, b));
  o[k++] = disk::length(a);
  put(o, k, disk::normalize(a));
  o[k++] = disk::inDisk(r, p) ? 1.0f : 0.0f;
  o[k++] = disk::diskTemperature(r, p);
  o[k++] = disk::orbitalSpeedLocal(r);
  const disk::Vec3 pos{r, 0.0f, 0.5f * r};
  const disk::Vec3 ray = disk::normalize(disk::Vec3{0.2f, -1.0f, 0.5f});
  o[k++] = disk::gFactor(r, c);
  o[k++] = disk::gFactorForRay(pos, ray);
  o[k++] = disk::intensityBoost(disk::gFactor(r, c));
  o[k++] = disk::radiantBrightness(disk::diskTemperature(r, p), p);
  put(o, k, disk::orbitalDirection(pos));
  put(o, k, disk::blackbodyColor(2000.0f + 3000.0f * i));
  put(o, k, disk::diskRadiance(r, c, p));
  const disk::Crossing x =
      disk::planeCrossing({0.0f, 1.0f + 0.1f * i, 0.0f}, {2.0f, -0.3f, 4.0f});
  o[k++] = x.hit ? 1.0f : 0.0f;
  o[k++] = x.t;
  put(o, k, x.pos);
  while (k < kFloatsPerCase) o[k++] = 0.0f;
}

__global__ void diskKernel(float* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < kCases) evalCase(i, out + i * kFloatsPerCase);
}

}  // namespace

TEST(DiskDeviceTest, DeviceMatchesHostAndKnownAnchors) {
  SKIP_IF_NO_CUDA_DEVICE();

  std::vector<float> host(kCases * kFloatsPerCase, 0.0f);
  for (int i = 0; i < kCases; ++i) evalCase(i, host.data() + i * kFloatsPerCase);

  float* dev = nullptr;
  ASSERT_EQ(cudaMalloc(&dev, host.size() * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMemset(dev, 0xFF, host.size() * sizeof(float)),
            cudaSuccess);  // NaN sentinel: an unwritten slot fails below
  diskKernel<<<1, 32>>>(dev);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<float> got(host.size());
  ASSERT_EQ(cudaMemcpy(got.data(), dev, got.size() * sizeof(float),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  cudaFree(dev);

  for (size_t n = 0; n < host.size(); ++n) {
    const float tol =
        1e-5f * std::fmax(std::fabs(host[n]), std::fabs(got[n])) + 1e-6f;
    EXPECT_LE(std::fabs(host[n] - got[n]), tol)
        << "case " << n / kFloatsPerCase << " slot " << n % kFloatsPerCase
        << " host=" << host[n] << " device=" << got[n];
  }

  // Anchors (case 0: r = 3, c = -1, inside the default disk): guards against
  // host and device agreeing on a wrong stub.
  const float* o = got.data();
  EXPECT_NEAR(o[18], 12000.0f, 0.2f);       // diskTemperature(3)
  EXPECT_NEAR(o[17], 1.0f, 1e-6f);          // inDisk(3)
  EXPECT_NEAR(o[19], 0.5f, 1e-6f);          // orbitalSpeedLocal(3)
  EXPECT_NEAR(o[20], 0.4714045f, 1e-5f);    // gFactor(3, -1)
  EXPECT_NEAR(o[22], 0.104757f, 1e-5f);     // intensityBoost(g)
  EXPECT_NEAR(o[23], 1.0f, 1e-5f);          // radiantBrightness at tInner
}
