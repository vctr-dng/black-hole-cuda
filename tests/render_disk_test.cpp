// Phase-4b tests: render() with config.disk = true.
//
// Kernel semantics under test (per pixel, sky/post at their defaults):
//   * primary ray = geo::makeRay(camera.pos, primaryRayDir); integrate with
//     geo::stepVerlet and dt = geo::adaptiveStep(|pos|, nearDisk = true)
//     (<= 0.085), rEscape = renderTraceParams(config).rEscape, maxSteps = 4000.
//   * after EACH step, disk::planeCrossing(posBefore, posAfter) (plane y = 0).
//     The first crossing whose position has r = |pos| in [diskInner,
//     diskOuter] terminates the ray (opaque disk) with colour
//       disk::diskRadiance(r, cosTheta, params),
//       cosTheta = dot(orbitalDirection(pos), -normalize(vel after the step)),
//     params = {diskInner, diskOuter, diskTempK, diskBrightness}. Crossings
//     outside the radial range are ignored.
//   * otherwise Horizon (r <= 1) and MaxSteps -> black, Escaped (r >= rEscape
//     and moving outward) -> skyGradient(normalize(vel)).
//   * bytes = lround(255 * clamp01(c)), alpha 255; no tone mapping.
//
// Orientation: orbitalDirection(pos) = normalize(z, 0, -x). Gas at world +x
// moves along -z, i.e. toward the camera (at -z looking +z). The camera's
// right vector is cross(up, forward) = cross((0,1,0),(0,0,1)) = +x, so the
// image-RIGHT half approaches and is Doppler-boosted, the left half recedes.
//
// Float tolerance: the CPU reference and the GPU kernel run the same float
// algorithm but with different FMA contraction / libm, and the disk radiance
// involves powf/logf. Disk bytes are mostly < 130, so +-3 bytes comfortably
// covers float noise; rays that graze the disk's rim or a step's plane
// crossing can flip class, hence the <= 1% class-mismatch allowance.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "blackhole/camera.hpp"
#include "blackhole/disk.hpp"
#include "blackhole/geodesic.hpp"
#include "blackhole/image.hpp"
#include "blackhole/render.hpp"
#include "cuda_test_support.hpp"

namespace {

using blackhole::Image;
using blackhole::RenderConfig;
namespace cam = blackhole::cam;
namespace geo = blackhole::geo;
namespace disk = blackhole::disk;

constexpr double kPi = 3.14159265358979323846;
constexpr double kBc = 2.598076211353316;  // 1.5 * sqrt(3)

enum class Kind { Shadow, Disk, Sky };

struct RefPixel {
  Kind kind = Kind::Shadow;
  std::uint8_t rgb[3] = {0, 0, 0};
};

RenderConfig makeConfig(int w, int h, float el, bool withDisk = true) {
  RenderConfig c;
  c.width = w;
  c.height = h;
  c.fovDegrees = 60.0f;
  c.elevationDegrees = el;
  c.observerDistance = 20.0f;
  c.disk = withDisk;
  return c;
}

std::uint8_t quantize(float c) {
  c = std::min(1.0f, std::max(0.0f, c));
  return static_cast<std::uint8_t>(std::lround(255.0f * c));
}

// Independent CPU reference of the kernel semantics in the file comment.
// cfg.disk = false ignores the disk (nearDisk steps are still used).
RefPixel refPixel(const RenderConfig& cfg, const cam::Camera& camera, int x,
                  int y) {
  const cam::Vec3 d = cam::primaryRayDir(camera, x, y);
  geo::Ray ray = geo::makeRay({camera.pos.x, camera.pos.y, camera.pos.z},
                              {d.x, d.y, d.z});
  const geo::TraceParams tp = blackhole::renderTraceParams(cfg);
  const disk::DiskParams dp{cfg.diskInner, cfg.diskOuter, cfg.diskTempK,
                            cfg.diskBrightness};
  RefPixel out;
  for (int i = 0; i < tp.maxSteps; ++i) {
    const geo::Vec3 p0 = ray.pos;
    geo::stepVerlet(ray, geo::adaptiveStep(geo::length(ray.pos), true));
    if (cfg.disk) {
      const disk::Crossing cr = disk::planeCrossing(
          {p0.x, p0.y, p0.z}, {ray.pos.x, ray.pos.y, ray.pos.z});
      if (cr.hit && disk::inDisk(disk::length(cr.pos), dp)) {
        const geo::Vec3 v = geo::normalize(ray.vel);
        const float cosT = disk::dot(disk::orbitalDirection(cr.pos),
                                     disk::Vec3{-v.x, -v.y, -v.z});
        const disk::Vec3 c =
            disk::diskRadiance(disk::length(cr.pos), cosT, dp);
        out.kind = Kind::Disk;
        out.rgb[0] = quantize(c.x);
        out.rgb[1] = quantize(c.y);
        out.rgb[2] = quantize(c.z);
        return out;
      }
    }
    const float rad = geo::length(ray.pos);
    if (rad <= geo::kRs) return out;  // Horizon -> black
    if (rad >= tp.rEscape && geo::dot(ray.pos, ray.vel) > 0.0f) {
      const geo::Vec3 v = geo::normalize(ray.vel);
      const cam::Vec3 c = cam::skyGradient({v.x, v.y, v.z});
      out.kind = Kind::Sky;
      out.rgb[0] = quantize(c.x);
      out.rgb[1] = quantize(c.y);
      out.rgb[2] = quantize(c.z);
      return out;
    }
  }
  return out;  // MaxSteps -> black
}

struct RefImage {
  int w = 0, h = 0;
  std::vector<RefPixel> px;
  const RefPixel& at(int x, int y) const { return px[y * w + x]; }
  std::size_t count(Kind k) const {
    std::size_t n = 0;
    for (const auto& p : px) n += p.kind == k;
    return n;
  }
};

RefImage refRender(const RenderConfig& cfg) {
  const cam::Camera camera =
      cam::makeCamera(cfg.width, cfg.height, cfg.observerDistance,
                      cfg.fovDegrees, cfg.elevationDegrees);
  RefImage r;
  r.w = cfg.width;
  r.h = cfg.height;
  r.px.resize(static_cast<std::size_t>(cfg.width) * cfg.height);
  for (int y = 0; y < cfg.height; ++y)
    for (int x = 0; x < cfg.width; ++x) r.px[y * cfg.width + x] = refPixel(cfg, camera, x, y);
  return r;
}

int ch(const Image& im, int x, int y, int c) {
  return im.data()[im.pixelIndex(x, y) + c];
}
bool isBlack(const Image& im, int x, int y) {
  return ch(im, x, y, 0) == 0 && ch(im, x, y, 1) == 0 && ch(im, x, y, 2) == 0;
}
int lum(const Image& im, int x, int y) {
  return ch(im, x, y, 0) + ch(im, x, y, 1) + ch(im, x, y, 2);
}

struct Agreement {
  std::size_t channels = 0, channelsOff = 0;
  std::size_t pixels = 0, classMismatch = 0;
};

// Class of a GPU pixel judged against the reference: a GPU pixel "matches" the
// reference class if its bytes are within +-3 of the reference colour.
Agreement compare(const Image& gpu, const RefImage& ref) {
  Agreement a;
  for (int y = 0; y < ref.h; ++y) {
    for (int x = 0; x < ref.w; ++x) {
      const RefPixel& r = ref.at(x, y);
      bool bad = false;
      for (int c = 0; c < 3; ++c) {
        ++a.channels;
        if (std::abs(ch(gpu, x, y, c) - static_cast<int>(r.rgb[c])) > 3) {
          ++a.channelsOff;
          bad = true;
        }
      }
      ++a.pixels;
      a.classMismatch += bad;
    }
  }
  return a;
}

void expectAgrees(const RenderConfig& cfg, const char* what) {
  const RefImage ref = refRender(cfg);
  const Image gpu = blackhole::render(cfg);
  const Agreement a = compare(gpu, ref);
  EXPECT_LE(a.channelsOff, a.channels * 2 / 100)
      << what << ": " << a.channelsOff << "/" << a.channels
      << " channels off by > 3 bytes";
  EXPECT_LE(a.classMismatch, a.pixels / 100)
      << what << ": " << a.classMismatch << "/" << a.pixels
      << " pixels differ from the reference";
}

// Pixels whose GPU colour differs from `base` by more than 8 on any channel.
std::size_t changedCount(const Image& a, const Image& b) {
  std::size_t n = 0;
  for (int y = 0; y < a.height(); ++y)
    for (int x = 0; x < a.width(); ++x) {
      bool ch8 = false;
      for (int c = 0; c < 3; ++c)
        ch8 = ch8 || std::abs(ch(a, x, y, c) - ch(b, x, y, c)) > 8;
      n += ch8;
    }
  return n;
}

// ---------------------------------------------------------------------------
// 1. CPU reference agreement
// ---------------------------------------------------------------------------

class DiskAgreement : public ::testing::TestWithParam<float> {};

TEST_P(DiskAgreement, MatchesCpuReference96x54) {
  SKIP_IF_NO_CUDA_DEVICE();
  const RenderConfig cfg = makeConfig(96, 54, GetParam());
  const RefImage ref = refRender(cfg);
  EXPECT_GT(ref.count(Kind::Disk), 300u) << "reference sees no disk";
  EXPECT_GT(ref.count(Kind::Shadow), 50u);
  EXPECT_GT(ref.count(Kind::Sky), 1000u);
  const Image gpu = blackhole::render(cfg);
  const Agreement a = compare(gpu, ref);
  EXPECT_LE(a.channelsOff, a.channels * 2 / 100)
      << a.channelsOff << "/" << a.channels << " channels off by > 3 bytes";
  EXPECT_LE(a.classMismatch, a.pixels / 100)
      << a.classMismatch << "/" << a.pixels << " pixels differ";
}

// The reference uses the nearDisk step cap (dt <= 0.085). The disk colour is
// smooth, so coarse steps (renderTraceParams' nearDisk = false, dt up to 0.7)
// change only the interpolated crossing slightly: the +-3 byte test above
// cannot see that, but a tight +-1 byte comparison can. The same-algorithm
// GPU render is within +-1 byte on all but a handful of rays (float noise on
// pixels sitting on a step boundary); coarse steps push ~0.6-0.8% of the
// pixels beyond +-1 byte, so we allow at most 0.25% (12 of 5184).
TEST_P(DiskAgreement, TightlyMatchesNearDiskStepReference) {
  SKIP_IF_NO_CUDA_DEVICE();
  const RenderConfig cfg = makeConfig(96, 54, GetParam());
  const RefImage ref = refRender(cfg);
  const Image gpu = blackhole::render(cfg);
  std::size_t off = 0, off3 = 0;
  for (int y = 0; y < 54; ++y)
    for (int x = 0; x < 96; ++x) {
      bool bad = false, bad3 = false;
      for (int c = 0; c < 3; ++c) {
        const int d = std::abs(ch(gpu, x, y, c) - int(ref.at(x, y).rgb[c]));
        bad = bad || d > 1;
        bad3 = bad3 || d > 3;
      }
      off += bad;
      off3 += bad3;
    }
  EXPECT_EQ(off3, 0u) << "pixels differing by more than 3 bytes";
  EXPECT_LE(off, 12u) << off << " of 5184 pixels differ by more than 1 byte";
}

INSTANTIATE_TEST_SUITE_P(Elevations, DiskAgreement,
                         ::testing::Values(10.0f, 30.0f, 60.0f));

TEST(RenderDiskTest, ThinAndOffsetDisksMatchReference) {
  SKIP_IF_NO_CUDA_DEVICE();
  // Narrow annulus: most disk-plane crossings fall OUTSIDE the radial range
  // and must be ignored (ray continues to the sky / far-side crossing).
  RenderConfig a = makeConfig(64, 36, 30.0f);
  a.diskInner = 4.0f;
  a.diskOuter = 6.0f;
  expectAgrees(a, "annulus 4..6");
  // Wide, hot, bright variant.
  RenderConfig b = makeConfig(64, 36, 15.0f);
  b.diskInner = 2.0f;
  b.diskOuter = 20.0f;
  b.diskTempK = 25000.0f;
  b.diskBrightness = 0.5f;
  expectAgrees(b, "wide hot 2..20");
  // Outer barely above inner still renders and matches.
  RenderConfig c = makeConfig(64, 36, 30.0f);
  c.diskOuter = c.diskInner + 0.5f;
  EXPECT_GT(refRender(c).count(Kind::Disk), 20u);
  expectAgrees(c, "narrow ring 3..3.5");
}

TEST(RenderDiskTest, AwkwardImageSizesMatchReference) {
  SKIP_IF_NO_CUDA_DEVICE();
  expectAgrees(makeConfig(45, 29, 30.0f), "45x29");
  expectAgrees(makeConfig(37, 23, 30.0f), "37x23");
}

// Edge-on (el = 0) and near face-on (el = 89.9) are degenerate for the disk
// plane test / the camera basis: only require sane output. Sky and shadow
// pixels (per the reference) must match, with <= 2% flips; disk pixels are not
// compared.
class DiskSmoke : public ::testing::TestWithParam<float> {};

TEST_P(DiskSmoke, RendersSanely) {
  SKIP_IF_NO_CUDA_DEVICE();
  const RenderConfig cfg = makeConfig(64, 36, GetParam());
  const Image gpu = blackhole::render(cfg);
  const RefImage ref = refRender(cfg);
  std::size_t checked = 0, bad = 0;
  for (int y = 0; y < 36; ++y)
    for (int x = 0; x < 64; ++x) {
      EXPECT_EQ(ch(gpu, x, y, 3), 255);
      const RefPixel& r = ref.at(x, y);
      if (r.kind == Kind::Disk) continue;
      ++checked;
      bool flip = false;
      for (int c = 0; c < 3; ++c)
        flip = flip || std::abs(ch(gpu, x, y, c) - int(r.rgb[c])) > 3;
      bad += flip;
    }
  EXPECT_GT(checked, 36u * 64u / 4);
  EXPECT_LE(bad * 50, checked) << bad << " of " << checked;
  EXPECT_TRUE(isBlack(gpu, 32, 18));
  EXPECT_TRUE(isBlack(gpu, 31, 17));
}

INSTANTIATE_TEST_SUITE_P(DegenerateElevations, DiskSmoke,
                         ::testing::Values(0.0f, 89.9f));

// ---------------------------------------------------------------------------
// 2. Structural properties (192x108)
// ---------------------------------------------------------------------------

TEST(RenderDiskTest, ShadowCentreIsBlackAndNotCoveredByDisk) {
  SKIP_IF_NO_CUDA_DEVICE();
  // The central ray (b ~ 0) heads straight at the hole and only crosses y = 0
  // at the origin (r ~ 0 < diskInner = 3): never a disk hit. The disk's inner
  // hole (r < 3) is wider than the shadow (b_c = 2.6 at D = 20), so the whole
  // central patch of radius 4 px (shadow radius ~ 12 px at 108 rows) is black.
  const RenderConfig cfg = makeConfig(192, 108, 30.0f);
  const Image gpu = blackhole::render(cfg);
  const RefImage ref = refRender(cfg);
  for (int dy = -4; dy <= 4; ++dy)
    for (int dx = -4; dx <= 4; ++dx) {
      if (dx * dx + dy * dy > 16) continue;
      const int x = 96 + dx, y = 54 + dy;
      EXPECT_TRUE(ref.at(x, y).kind == Kind::Shadow) << x << "," << y;
      EXPECT_TRUE(isBlack(gpu, x, y)) << "GPU not black at " << x << "," << y;
    }
}

TEST(RenderDiskTest, DiskPixelsExistAndChangeTheImage) {
  SKIP_IF_NO_CUDA_DEVICE();
  const RenderConfig cfg = makeConfig(192, 108, 30.0f);
  const RefImage ref = refRender(cfg);
  EXPECT_GT(ref.count(Kind::Disk), 800u);
  const Image on = blackhole::render(cfg);
  const Image off = blackhole::render(makeConfig(192, 108, 30.0f, false));
  EXPECT_GT(changedCount(on, off), 800u)
      << "disk=true render barely differs from disk=false";
}

TEST(RenderDiskTest, ApproachingSideIsImageRightAndMuchBrighter) {
  SKIP_IF_NO_CUDA_DEVICE();
  // Doppler: at the inner edge v = sqrt(0.5/(r-1)) = 0.5 at r = 3, so
  // g ~ 1/(1 -+ 0.5 cos) times sqrt((r-1.5)/r)*... and intensity ~ g^3 * T^4:
  // approaching vs receding differ by ~ ((1+v)/(1-v))^3 ~ 27x at the extremes.
  // Averaged over the whole disk (inclination 30 deg, all azimuths, cos
  // factor) the reference gives a right/left mean ratio ~3.3 (3.8 at el 10);
  // we require >= 2, which excludes g = 1 (ratio 1) and a flipped orbit
  // direction (ratio < 1).
  for (float el : {10.0f, 30.0f}) {
    const RenderConfig cfg = makeConfig(192, 108, el);
    const RefImage ref = refRender(cfg);
    const Image gpu = blackhole::render(cfg);
    double sum[2] = {0, 0};
    int n[2] = {0, 0};
    for (int y = 0; y < ref.h; ++y)
      for (int x = 0; x < ref.w; ++x)
        if (ref.at(x, y).kind == Kind::Disk) {
          const int side = x >= ref.w / 2 ? 1 : 0;
          sum[side] += lum(gpu, x, y);
          ++n[side];
        }
    ASSERT_GT(n[0], 300) << "el " << el;
    ASSERT_GT(n[1], 300) << "el " << el;
    const double meanL = sum[0] / n[0], meanR = sum[1] / n[1];
    EXPECT_GT(meanL, 0.0) << "el " << el;
    EXPECT_GE(meanR, 2.0 * meanL) << "el " << el << " meanL " << meanL
                                  << " meanR " << meanR;
    EXPECT_LE(meanR, 8.0 * meanL) << "el " << el << " (sanity upper bound)";
  }
}

TEST(RenderDiskTest, LensedFarSideImageAppearsAboveTheShadow) {
  SKIP_IF_NO_CUDA_DEVICE();
  // At el = 10 the disk is nearly edge-on; the far side of the disk is bent
  // over the top of the hole. Shadow radius ~ 12.25 px at 108 rows, so rows
  // above cy - 14 = 40 cannot contain the shadow, and the primary (un-lensed)
  // disk, an ellipse of semi-minor axis ~ D-projected 12*sin(10deg) ~ 2 rs,
  // reaches only ~ 11 px above the centre line. The reference arch rises to
  // ~ 22 px above the centre.
  const RenderConfig cfg = makeConfig(192, 108, 10.0f);
  const RefImage ref = refRender(cfg);
  const Image gpu = blackhole::render(cfg);
  RenderConfig noDisk = cfg;
  noDisk.disk = false;
  const RefImage plain = refRender(noDisk);  // same nearDisk steps, no disk
  // Rows 0..39 (>= 14 px above the centre row) are above the shadow.
  int arch = 0, archNonSky = 0;
  for (int y = 0; y < 54 - 14; ++y)
    for (int x = 0; x < ref.w; ++x)
      if (ref.at(x, y).kind == Kind::Disk) {
        ++arch;
        bool differs = false;
        for (int c = 0; c < 3; ++c)
          differs = differs ||
                    std::abs(ch(gpu, x, y, c) - plain.at(x, y).rgb[c]) > 3;
        archNonSky += differs;
      }
  ASSERT_GT(arch, 150) << "reference has no lensed arch above the shadow";
  EXPECT_GE(archNonSky, arch * 95 / 100)
      << "GPU pixels in the arch region look like plain sky";
}

TEST(RenderDiskTest, FaceOnRingProfileAlongTheApproachingAxis) {
  SKIP_IF_NO_CUDA_DEVICE();
  // At el = 60 walking right from the image centre along the middle row the
  // classes must run Shadow+ Sky+ (the hole between shadow and inner edge)
  // Disk+ Sky+ (beyond the outer edge), and the disk brightness falls with
  // radius (T^4 ~ r^-3) so the first quarter of the disk run is much brighter
  // than its last quarter.
  const RenderConfig cfg = makeConfig(192, 108, 60.0f);
  const RefImage ref = refRender(cfg);
  const Image gpu = blackhole::render(cfg);
  const int y = 54;
  // Encode the row as a string of S(hadow) K(sky) D(isk). Walking right from
  // the centre: a shadow run, then sky/disk (lensed secondary images make the
  // hole edge ragged, so single K/D flips are allowed there), then a long sky
  // tail. Never a shadow pixel after the first non-shadow pixel.
  std::string row;
  std::vector<int> diskX;
  for (int x = 96; x < 192; ++x) {
    const Kind k = ref.at(x, y).kind;
    row += k == Kind::Shadow ? 'S' : k == Kind::Disk ? 'D' : 'K';
    if (k == Kind::Disk) diskX.push_back(x);
  }
  EXPECT_GE(row.find_first_not_of('S'), 6u) << row;  // shadow run at centre
  EXPECT_EQ(row.find('S', row.find_first_not_of('S')), std::string::npos)
      << row;  // no shadow after it
  EXPECT_EQ(row.substr(row.size() - 15), std::string(15, 'K')) << row;
  EXPECT_NE(row.find("DDDDDDDDDDDDDDDDDDDD"), std::string::npos) << row;
  ASSERT_GT(diskX.size(), 20u);
  const std::size_t q = diskX.size() / 4;
  double first = 0, last = 0;
  for (std::size_t i = 0; i < q; ++i) {
    first += lum(gpu, diskX[i], y);
    last += lum(gpu, diskX[diskX.size() - 1 - i], y);
  }
  EXPECT_GT(first, 3.0 * last) << "first " << first << " last " << last;
  // Inside the hole the GPU shows no disk: the shadow pixels are black.
  EXPECT_TRUE(isBlack(gpu, 96, y));
  // GPU disk pixels on the run agree with the reference.
  for (int x : diskX)
    for (int c = 0; c < 3; ++c)
      EXPECT_NEAR(ch(gpu, x, y, c), ref.at(x, y).rgb[c], 4.0) << x;
}

// ---------------------------------------------------------------------------
// 3. Parameter effects
// ---------------------------------------------------------------------------

TEST(RenderDiskTest, SmallerOuterRadiusGivesFewerDiskPixels) {
  SKIP_IF_NO_CUDA_DEVICE();
  const Image base = blackhole::render(makeConfig(96, 54, 30.0f, false));
  std::size_t prev = 0;
  std::size_t prevRef = 0;
  for (float outer : {5.0f, 8.0f, 12.0f}) {
    RenderConfig cfg = makeConfig(96, 54, 30.0f);
    cfg.diskOuter = outer;
    const std::size_t changed = changedCount(blackhole::render(cfg), base);
    const std::size_t refDisk = refRender(cfg).count(Kind::Disk);
    EXPECT_GT(changed, prev) << "outer " << outer;
    EXPECT_GT(refDisk, prevRef) << "outer " << outer;
    EXPECT_GT(changed, refDisk * 9 / 10) << "outer " << outer;
    prev = changed;
    prevRef = refDisk;
  }
  EXPECT_GT(prevRef, 2000u);
}

TEST(RenderDiskTest, LargerInnerRadiusEnlargesTheHole) {
  SKIP_IF_NO_CUDA_DEVICE();
  const Image base = blackhole::render(makeConfig(96, 54, 60.0f, false));
  std::size_t prev = 1u << 30;
  int prevFirst = 0;
  for (float inner : {3.0f, 5.0f, 7.0f}) {
    RenderConfig cfg = makeConfig(96, 54, 60.0f);
    cfg.diskInner = inner;
    const Image gpu = blackhole::render(cfg);
    const std::size_t changed = changedCount(gpu, base);
    EXPECT_LT(changed, prev) << "inner " << inner;
    prev = changed;
    // First non-sky-like pixel along the right axis moves outward: the pixel
    // at 0.9*inner*(pixels per unit) is still hole. Check via the reference
    // that centre-row hole width (non-disk run after the centre) grows, and
    // that the GPU agrees.
    const RefImage ref = refRender(cfg);
    int firstDisk = -1;
    for (int x = 48; x < 96; ++x)
      if (ref.at(x, 27).kind == Kind::Disk) { firstDisk = x; break; }
    ASSERT_GT(firstDisk, 0) << "inner " << inner;
    EXPECT_GT(firstDisk, prevFirst) << "inner " << inner;
    prevFirst = firstDisk;
    EXPECT_GT(lum(gpu, firstDisk, 27), 0) << "inner " << inner;
    const Agreement a = compare(gpu, ref);
    EXPECT_LE(a.classMismatch, a.pixels / 100) << "inner " << inner;
  }
}

TEST(RenderDiskTest, ZeroBrightnessDiskIsBlackAndSkyUntouched) {
  SKIP_IF_NO_CUDA_DEVICE();
  const RenderConfig bright = makeConfig(96, 54, 30.0f);
  RenderConfig dark = bright;
  dark.diskBrightness = 0.0f;
  const RefImage ref = refRender(bright);
  const Image a = blackhole::render(bright);
  const Image b = blackhole::render(dark);
  // Count violations (<= 0.2% of pixels tolerated: a valid implementation may
  // flip a boundary ray's class relative to the CPU reference).
  std::size_t diskPx = 0, bad = 0;
  for (int y = 0; y < 54; ++y)
    for (int x = 0; x < 96; ++x) {
      if (ref.at(x, y).kind == Kind::Disk) {
        ++diskPx;
        bad += !isBlack(b, x, y);
      } else {
        for (int c = 0; c < 3; ++c)
          if (ch(a, x, y, c) != ch(b, x, y, c)) { ++bad; break; }
      }
    }
  EXPECT_GT(diskPx, 1500u);
  EXPECT_LE(bad, 5184u * 2 / 1000) << bad << " violating pixels";
}

TEST(RenderDiskTest, DoubleBrightnessScalesLinearlyUntilSaturation) {
  SKIP_IF_NO_CUDA_DEVICE();
  const RenderConfig one = makeConfig(96, 54, 30.0f);
  RenderConfig two = one;
  two.diskBrightness = 2.0f;
  const RefImage ref = refRender(one);
  const Image a = blackhole::render(one);
  const Image b = blackhole::render(two);
  // Violations are counted per pixel (<= 0.2% tolerated, see above).
  std::size_t linear = 0, brighter = 0, bad = 0;
  for (int y = 0; y < 54; ++y)
    for (int x = 0; x < 96; ++x) {
      bool pxBad = false;
      for (int c = 0; c < 3; ++c) {
        const int v1 = ch(a, x, y, c), v2 = ch(b, x, y, c);
        if (v2 < v1) pxBad = true;
        if (ref.at(x, y).kind != Kind::Disk) {
          if (v1 != v2) pxBad = true;
          continue;
        }
        if (v1 > 0 && v1 <= 120) {
          // 255*2c vs 2*(255*c): rounding differs by at most 1.
          if (std::abs(v2 - 2 * v1) > 1) pxBad = true;
          ++linear;
        }
        brighter += v2 > v1;
      }
      bad += pxBad;
    }
  EXPECT_GT(linear, 2000u);
  EXPECT_GT(brighter, 2000u);
  EXPECT_LE(bad, 5184u * 2 / 1000) << bad << " violating pixels";
}

TEST(RenderDiskTest, HotterDiskIsBluer) {
  SKIP_IF_NO_CUDA_DEVICE();
  // Colour = blackbodyColor(g * T(r)) * intensity; the LUT chromaticity
  // blue/red grows monotonically with temperature (b/r = 0.87 at 6000 K,
  // 2.5 at 20000 K for g = 1). Only pixels unsaturated (all channels <= 250)
  // and not too dark (red >= 20) in BOTH renders are compared, so the byte
  // ratio equals the chromaticity ratio. The disk footprint (which
  // pixels are disk) does not depend on temperature.
  RenderConfig cold = makeConfig(96, 54, 30.0f);
  cold.diskTempK = 6000.0f;
  cold.diskBrightness = 1.0f;
  RenderConfig hot = cold;
  hot.diskTempK = 20000.0f;
  const RefImage ref = refRender(cold);
  const Image c = blackhole::render(cold);
  const Image h = blackhole::render(hot);
  double sumCold = 0, sumHot = 0;
  int n = 0, bluer = 0;
  for (int y = 0; y < 54; ++y)
    for (int x = 0; x < 96; ++x) {
      if (ref.at(x, y).kind != Kind::Disk) continue;
      const int rc = ch(c, x, y, 0), bc = ch(c, x, y, 2);
      const int rh = ch(h, x, y, 0), bh = ch(h, x, y, 2);
      if (rc < 20 || rh < 20) continue;
      if (std::max({rc, ch(c, x, y, 1), bc, rh, ch(h, x, y, 1), bh}) > 250) continue;
      const double ratioC = double(bc) / rc, ratioH = double(bh) / rh;
      ++n;
      bluer += ratioH > ratioC;
      sumCold += ratioC;
      sumHot += ratioH;
    }
  ASSERT_GT(n, 100);
  EXPECT_GE(bluer, n * 95 / 100);
  EXPECT_GT(sumHot / n, 1.5 * (sumCold / n))
      << "mean b/r cold " << sumCold / n << " hot " << sumHot / n;
}

// ---------------------------------------------------------------------------
// 4. disk = false and artefact checks
// ---------------------------------------------------------------------------

TEST(RenderDiskTest, DiskOffIgnoresDiskParametersBitwise) {
  SKIP_IF_NO_CUDA_DEVICE();
  const RenderConfig base = makeConfig(64, 36, 30.0f, false);
  RenderConfig odd = base;
  odd.diskInner = 5.0f;
  odd.diskOuter = 9.0f;
  odd.diskTempK = 30000.0f;
  odd.diskBrightness = 7.0f;
  const Image a = blackhole::render(base);
  const Image b = blackhole::render(odd);
  const Image d = blackhole::render([] {
    RenderConfig c;
    c.width = 64;
    c.height = 36;
    c.fovDegrees = 60.0f;
    c.observerDistance = 20.0f;
    c.elevationDegrees = 30.0f;
    c.disk = false;
    return c;
  }());
  ASSERT_EQ(a.size(), b.size());
  EXPECT_EQ(0, std::memcmp(a.data(), b.data(), a.size()));
  EXPECT_EQ(0, std::memcmp(a.data(), d.data(), a.size()));
}

TEST(RenderDiskTest, NoArtefactsOutsideTheDiskImage) {
  SKIP_IF_NO_CUDA_DEVICE();
  // Pixels the reference classes as Sky must be sky-coloured (+-3), pixels it
  // classes as Shadow must be black: fewer than 1% violations each.
  for (float el : {10.0f, 30.0f, 60.0f}) {
    const RenderConfig cfg = makeConfig(96, 54, el);
    const RefImage ref = refRender(cfg);
    const Image gpu = blackhole::render(cfg);
    std::size_t skyBad = 0, shadowBad = 0, sky = 0, shadow = 0;
    for (int y = 0; y < 54; ++y)
      for (int x = 0; x < 96; ++x) {
        const RefPixel& r = ref.at(x, y);
        if (r.kind == Kind::Sky) {
          ++sky;
          bool bad = false;
          for (int c = 0; c < 3; ++c)
            bad = bad || std::abs(ch(gpu, x, y, c) - int(r.rgb[c])) > 3;
          skyBad += bad;
        } else if (r.kind == Kind::Shadow) {
          ++shadow;
          shadowBad += !isBlack(gpu, x, y);
        }
      }
    EXPECT_LT(skyBad * 100, sky) << "el " << el;
    EXPECT_LT(shadowBad * 100, shadow) << "el " << el;
  }
}

// ---------------------------------------------------------------------------
// 5. Determinism
// ---------------------------------------------------------------------------

TEST(RenderDiskTest, RenderIsDeterministic) {
  SKIP_IF_NO_CUDA_DEVICE();
  const RenderConfig cfg = makeConfig(96, 54, 30.0f);
  const Image a = blackhole::render(cfg);
  const Image b = blackhole::render(cfg);
  ASSERT_EQ(a.size(), b.size());
  EXPECT_EQ(0, std::memcmp(a.data(), b.data(), a.size()));
}

TEST(RenderDiskTest, AlphaIsOpaqueEverywhere) {
  SKIP_IF_NO_CUDA_DEVICE();
  const Image a = blackhole::render(makeConfig(64, 36, 30.0f));
  for (int y = 0; y < 36; ++y)
    for (int x = 0; x < 64; ++x) ASSERT_EQ(ch(a, x, y, 3), 255);
}

}  // namespace
