// CPU-only tests of the Phase-1 camera (include/blackhole/camera.hpp).
//
// Convention under test (rs = 1, black hole at the world origin):
//   world axes: x right, y up, z forward. A camera at elevation 0 sits at
//   (0, 0, -D) looking toward +z. Image row 0 is the TOP of the image, pixel
//   centers are at (px + 0.5, py + 0.5).
//
// Most tests deliberately use an *independent oracle* (plain double math in
// this file) instead of re-deriving expected values with camera.hpp itself.
//
// Tolerances: everything is scalar float, so one operation carries ~6e-8
// relative error; a normalized vector is accurate to a few ulp (~2e-7). We use
// 1e-6 for "should be exactly this unit vector / angle-free" checks, 1e-5 where
// an atan/tan is involved or a projective ratio (components up to ~10) is
// compared, and 1e-5 absolute on positions of magnitude ~20 (20 * 6e-8 * a
// few).

#include "blackhole/camera.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <initializer_list>
#include <vector>

namespace {

using blackhole::cam::Camera;
using blackhole::cam::makeCamera;
using blackhole::cam::primaryRayDir;
using blackhole::cam::primaryRayDirSub;
using blackhole::cam::skyGradient;
using blackhole::cam::Vec3;

constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg = kPi / 180.0;

// --- independent double-precision helpers (do not use camera.hpp) -----------

double dot3(const Vec3& a, const Vec3& b) {
  return static_cast<double>(a.x) * b.x + static_cast<double>(a.y) * b.y +
         static_cast<double>(a.z) * b.z;
}

double len3(const Vec3& a) { return std::sqrt(dot3(a, a)); }

double crossLen3(const Vec3& a, const Vec3& b) {
  const double cx =
      static_cast<double>(a.y) * b.z - static_cast<double>(a.z) * b.y;
  const double cy =
      static_cast<double>(a.z) * b.x - static_cast<double>(a.x) * b.z;
  const double cz =
      static_cast<double>(a.x) * b.y - static_cast<double>(a.y) * b.x;
  return std::sqrt(cx * cx + cy * cy + cz * cz);
}

// Angle between two vectors; atan2 form is accurate for tiny angles too.
double angleBetween(const Vec3& a, const Vec3& b) {
  return std::atan2(crossLen3(a, b), dot3(a, b));
}

constexpr float kElevations[] = {0.0f, 30.0f, 60.0f, 89.9f};

}  // namespace

// --- Vec3 helpers
// -------------------------------------------------------------

TEST(CameraVec3Test, ArithmeticDotCrossLengthNormalize) {
  const Vec3 a{1.0f, 2.0f, 3.0f};
  const Vec3 b{-4.0f, 5.0f, 0.5f};

  const Vec3 sum = a + b;
  EXPECT_FLOAT_EQ(sum.x, -3.0f);
  EXPECT_FLOAT_EQ(sum.y, 7.0f);
  EXPECT_FLOAT_EQ(sum.z, 3.5f);

  const Vec3 diff = a - b;
  EXPECT_FLOAT_EQ(diff.x, 5.0f);
  EXPECT_FLOAT_EQ(diff.y, -3.0f);
  EXPECT_FLOAT_EQ(diff.z, 2.5f);

  const Vec3 scaled = a * 2.0f;
  EXPECT_FLOAT_EQ(scaled.x, 2.0f);
  EXPECT_FLOAT_EQ(scaled.y, 4.0f);
  EXPECT_FLOAT_EQ(scaled.z, 6.0f);

  EXPECT_FLOAT_EQ(blackhole::cam::dot(a, b), -4.0f + 10.0f + 1.5f);

  // Standard right-handed cross product: x cross y = z.
  const Vec3 xy = blackhole::cam::cross(Vec3{1, 0, 0}, Vec3{0, 1, 0});
  EXPECT_FLOAT_EQ(xy.x, 0.0f);
  EXPECT_FLOAT_EQ(xy.y, 0.0f);
  EXPECT_FLOAT_EQ(xy.z, 1.0f);
  const Vec3 c = blackhole::cam::cross(a, b);
  EXPECT_FLOAT_EQ(c.x, 2.0f * 0.5f - 3.0f * 5.0f);
  EXPECT_FLOAT_EQ(c.y, 3.0f * -4.0f - 1.0f * 0.5f);
  EXPECT_FLOAT_EQ(c.z, 1.0f * 5.0f - 2.0f * -4.0f);

  EXPECT_FLOAT_EQ(blackhole::cam::length(Vec3{3.0f, 4.0f, 12.0f}), 13.0f);

  const Vec3 n = blackhole::cam::normalize(Vec3{0.0f, 3.0f, 4.0f});
  EXPECT_NEAR(n.x, 0.0f, 1e-7);
  EXPECT_NEAR(n.y, 0.6f, 1e-6);
  EXPECT_NEAR(n.z, 0.8f, 1e-6);
}

// --- makeCamera
// ---------------------------------------------------------------

TEST(MakeCameraTest, EdgeOnCameraSitsOnMinusZAxisLookingAlongPlusZ) {
  const Camera c = makeCamera(64, 48, 20.0f, 60.0f, 0.0f);

  EXPECT_NEAR(c.pos.x, 0.0f, 1e-6);
  EXPECT_NEAR(c.pos.y, 0.0f, 1e-6);
  EXPECT_NEAR(c.pos.z, -20.0f, 1e-5);

  EXPECT_NEAR(c.forward.x, 0.0f, 1e-6);
  EXPECT_NEAR(c.forward.y, 0.0f, 1e-6);
  EXPECT_NEAR(c.forward.z, 1.0f, 1e-6);

  // world x is image-right, world y is image-up
  EXPECT_NEAR(c.right.x, 1.0f, 1e-6);
  EXPECT_NEAR(c.right.y, 0.0f, 1e-6);
  EXPECT_NEAR(c.right.z, 0.0f, 1e-6);

  EXPECT_NEAR(c.up.x, 0.0f, 1e-6);
  EXPECT_NEAR(c.up.y, 1.0f, 1e-6);
  EXPECT_NEAR(c.up.z, 0.0f, 1e-6);
}

TEST(MakeCameraTest, PositionMatchesElevationFormula) {
  const float distance = 17.5f;
  for (const float el : kElevations) {
    SCOPED_TRACE(el);
    const Camera c = makeCamera(32, 32, distance, 60.0f, el);
    const double s = std::sin(el * kDeg);
    const double co = std::cos(el * kDeg);
    EXPECT_NEAR(c.pos.x, 0.0, 1e-5);
    EXPECT_NEAR(c.pos.y, distance * s, 1e-5);
    EXPECT_NEAR(c.pos.z, -distance * co, 1e-5);
    EXPECT_NEAR(len3(c.pos), distance, 1e-5);
    // Positive elevation = above the disk plane (y = 0).
    if (el > 0.0f) EXPECT_GT(c.pos.y, 0.0f);
  }
}

TEST(MakeCameraTest, ForwardIsUnitAndPointsAtBlackHoleOrigin) {
  for (const float el : kElevations) {
    SCOPED_TRACE(el);
    const Camera c = makeCamera(32, 32, 25.0f, 60.0f, el);
    EXPECT_NEAR(len3(c.forward), 1.0, 1e-6);
    // forward == (origin - pos) / |pos|
    const double d = len3(c.pos);
    EXPECT_NEAR(c.forward.x, -c.pos.x / d, 1e-6);
    EXPECT_NEAR(c.forward.y, -c.pos.y / d, 1e-6);
    EXPECT_NEAR(c.forward.z, -c.pos.z / d, 1e-6);
    // Analytic: forward = (0, -sin(el), cos(el)).
    EXPECT_NEAR(c.forward.x, 0.0, 1e-6);
    EXPECT_NEAR(c.forward.y, -std::sin(el * kDeg), 1e-6);
    EXPECT_NEAR(c.forward.z, std::cos(el * kDeg), 1e-6);
  }
}

TEST(MakeCameraTest, BasisIsOrthonormalAndRightHanded) {
  for (const float el : kElevations) {
    SCOPED_TRACE(el);
    const Camera c = makeCamera(32, 32, 25.0f, 60.0f, el);
    EXPECT_NEAR(len3(c.right), 1.0, 1e-6);
    EXPECT_NEAR(len3(c.up), 1.0, 1e-6);
    EXPECT_NEAR(dot3(c.right, c.up), 0.0, 1e-6);
    EXPECT_NEAR(dot3(c.right, c.forward), 0.0, 1e-6);
    EXPECT_NEAR(dot3(c.up, c.forward), 0.0, 1e-6);

    // Handedness: right x up == forward (x right, y up, z forward is
    // right-handed under the standard cross product). A mirrored camera
    // (right negated) would give -forward here.
    const double cx = static_cast<double>(c.right.y) * c.up.z -
                      static_cast<double>(c.right.z) * c.up.y;
    const double cy = static_cast<double>(c.right.z) * c.up.x -
                      static_cast<double>(c.right.x) * c.up.z;
    const double cz = static_cast<double>(c.right.x) * c.up.y -
                      static_cast<double>(c.right.y) * c.up.x;
    EXPECT_NEAR(cx, c.forward.x, 1e-6);
    EXPECT_NEAR(cy, c.forward.y, 1e-6);
    EXPECT_NEAR(cz, c.forward.z, 1e-6);
  }
}

TEST(MakeCameraTest, RightIsHorizontalAndUpPointsTowardWorldUp) {
  for (const float el : kElevations) {
    SCOPED_TRACE(el);
    const Camera c = makeCamera(32, 32, 25.0f, 60.0f, el);
    // right = normalize(cross(world_up, forward)) has no y component and is
    // +x for every elevation in [0, 90) (forward.z > 0).
    EXPECT_NEAR(c.right.y, 0.0f, 1e-7);
    EXPECT_NEAR(c.right.x, 1.0f, 1e-6);
    EXPECT_NEAR(c.right.z, 0.0f, 1e-6);
    // up tilts with the camera but always keeps a positive world-y component
    // (the image top is never below the horizon), equal to cos(el).
    EXPECT_GT(c.up.y, 0.0f);
    EXPECT_NEAR(c.up.y, std::cos(el * kDeg), 1e-6);
    EXPECT_NEAR(c.up.z, std::sin(el * kDeg), 1e-6);
  }
}

TEST(MakeCameraTest, NearPoleElevationStillWellDefined) {
  // el = 89.9 degrees: forward is almost straight down, but right/up must stay
  // finite, unit and orthogonal.
  const Camera c = makeCamera(32, 32, 20.0f, 60.0f, 89.9f);
  ASSERT_TRUE(std::isfinite(c.right.x) && std::isfinite(c.up.y));
  EXPECT_NEAR(c.right.x, 1.0f, 1e-5);
  EXPECT_GT(c.up.y, 0.0f);
  EXPECT_LT(c.up.y, 0.01f);  // cos(89.9 deg) = 1.7e-3
  EXPECT_NEAR(c.forward.y, -std::sin(89.9 * kDeg), 1e-6);
}

TEST(MakeCameraTest, StoresFovAspectAndImageSize) {
  const Camera c = makeCamera(16, 9, 20.0f, 60.0f, 10.0f);
  EXPECT_EQ(c.width, 16);
  EXPECT_EQ(c.height, 9);
  // Floating-point (not integer) division: 16/9 = 1.777..., not 1.
  EXPECT_NEAR(c.aspect, 16.0 / 9.0, 1e-6);
  // fovY is the FULL vertical fov; tanHalfFovY = tan(fov / 2).
  EXPECT_NEAR(c.tanHalfFovY, std::tan(30.0 * kDeg), 1e-6);

  EXPECT_NEAR(makeCamera(8, 8, 20.0f, 90.0f, 0.0f).tanHalfFovY, 1.0, 1e-6);
  EXPECT_NEAR(makeCamera(8, 8, 20.0f, 120.0f, 0.0f).tanHalfFovY,
              std::tan(60.0 * kDeg), 1e-5);
  EXPECT_NEAR(makeCamera(5, 20, 20.0f, 60.0f, 0.0f).aspect, 0.25, 1e-7);
}

// --- primary rays
// -------------------------------------------------------------

TEST(PrimaryRayTest, HandBuiltCameraPinsAxisAndRowConventions) {
  // Independent of makeCamera: identity-axes camera, 90 degree vertical fov
  // (tanHalfFovY = 1), aspect 2, 4x2 image.
  Camera c;
  c.pos = {0.0f, 0.0f, -10.0f};
  c.forward = {0.0f, 0.0f, 1.0f};
  c.right = {1.0f, 0.0f, 0.0f};
  c.up = {0.0f, 1.0f, 0.0f};
  c.tanHalfFovY = 1.0f;
  c.aspect = 2.0f;
  c.width = 4;
  c.height = 2;

  // Un-normalized directions: forward + ndcX*aspect*t*right + ndcY*t*up with
  // ndcX = 2(px+.5)/4 - 1 in {-.75,-.25,.25,.75}, ndcY = 1 - 2(py+.5)/2 in
  // {+.5 (top row), -.5 (bottom row)}.
  struct Case {
    int px, py;
    double ux, uy, uz;
  };
  const Case cases[] = {
      {0, 0, -1.5, +0.5, 1.0},  // top-left: left = -x, top = +y
      {3, 0, +1.5, +0.5, 1.0},  // top-right
      {0, 1, -1.5, -0.5, 1.0},  // bottom-left
      {3, 1, +1.5, -0.5, 1.0},  // bottom-right
      {1, 0, -0.5, +0.5, 1.0}, {2, 1, +0.5, -0.5, 1.0},
  };
  for (const Case& k : cases) {
    SCOPED_TRACE(::testing::Message() << "px=" << k.px << " py=" << k.py);
    const double n = std::sqrt(k.ux * k.ux + k.uy * k.uy + k.uz * k.uz);
    const Vec3 d = primaryRayDir(c, k.px, k.py);
    EXPECT_NEAR(d.x, k.ux / n, 1e-6);
    EXPECT_NEAR(d.y, k.uy / n, 1e-6);
    EXPECT_NEAR(d.z, k.uz / n, 1e-6);
  }
}

TEST(PrimaryRayTest, CenterPixelOfOddImagePointsAlongForward) {
  for (const float el : kElevations) {
    SCOPED_TRACE(el);
    const Camera c = makeCamera(31, 17, 20.0f, 70.0f, el);
    const Vec3 d = primaryRayDir(c, 15, 8);
    ASSERT_NEAR(len3(c.forward), 1.0, 1e-6);  // guards against 0 == 0
    ASSERT_NEAR(len3(d), 1.0, 1e-6);
    // ndc = 0 exactly, so the un-normalized ray is exactly `forward`.
    EXPECT_NEAR(d.x, c.forward.x, 1e-6);
    EXPECT_NEAR(d.y, c.forward.y, 1e-6);
    EXPECT_NEAR(d.z, c.forward.z, 1e-6);
  }
}

TEST(PrimaryRayTest, CenterRayPassesThroughBlackHoleOrigin) {
  // The ray from the camera through the image center must hit the origin:
  // its perpendicular distance from the origin is |pos x dir| (dir is unit).
  for (const float el : kElevations) {
    SCOPED_TRACE(el);
    const Camera c = makeCamera(31, 17, 20.0f, 70.0f, el);
    const Vec3 d = primaryRayDir(c, 15, 8);
    EXPECT_NEAR(crossLen3(c.pos, d), 0.0, 2e-5);  // |pos| ~ 20 -> 20 * 1e-6
    // ... and points toward the origin, not away from it.
    EXPECT_GT(-dot3(c.pos, d), 19.9);
  }
}

TEST(PrimaryRayTest, EdgeOnCameraRightPixelsLookRightAndTopPixelsLookUp) {
  const Camera c = makeCamera(9, 7, 20.0f, 60.0f, 0.0f);
  EXPECT_GT(primaryRayDir(c, 8, 3).x, 0.0f);  // right column -> +x
  EXPECT_LT(primaryRayDir(c, 0, 3).x, 0.0f);  // left column -> -x
  EXPECT_GT(primaryRayDir(c, 4, 0).y, 0.0f);  // top row -> +y (row 0 = top)
  EXPECT_LT(primaryRayDir(c, 4, 6).y, 0.0f);  // bottom row -> -y
}

TEST(PrimaryRayTest, AllDirectionsAreUnitLength) {
  struct Setup {
    int w, h;
    float fov, el;
  };
  const Setup setups[] = {{16, 9, 60.0f, 0.0f},   {7, 5, 1.0f, 30.0f},
                          {5, 12, 120.0f, 60.0f}, {31, 17, 170.0f, 89.9f},
                          {1, 1, 60.0f, 10.0f},   {64, 48, 90.0f, 45.0f}};
  for (const Setup& s : setups) {
    SCOPED_TRACE(::testing::Message()
                 << s.w << "x" << s.h << " fov=" << s.fov << " el=" << s.el);
    const Camera c = makeCamera(s.w, s.h, 20.0f, s.fov, s.el);
    for (int y = 0; y < s.h; ++y) {
      for (int x = 0; x < s.w; ++x) {
        const Vec3 d = primaryRayDir(c, x, y);
        ASSERT_NEAR(len3(d), 1.0, 1e-6) << "px=" << x << " py=" << y;
      }
    }
  }
}

TEST(PrimaryRayTest, ProjectiveOracleForEveryPixel) {
  // For a unit ray d the camera-space ratios
  //   dot(d,right)/dot(d,forward) = ndcX * aspect * tan(fovY/2)
  //   dot(d,up)   /dot(d,forward) = ndcY * tan(fovY/2)
  // are independent of normalization, handedness bugs, flipped rows, corner vs
  // center sampling and full-vs-half fov. Check them against plain double math
  // using only the camera's *basis* (so this test fails for ray bugs, while the
  // makeCamera tests above catch basis bugs). Error: ratio of two O(1) float
  // dot products, magnitude <= ~1.5 for fov 60 -> ~1e-6; 1e-5 is safe.
  const int w = 13;
  const int h = 8;  // even height: no pixel row exactly on the axis
  const float fov = 60.0f;
  const double t = std::tan(0.5 * fov * kDeg);
  const double aspect = static_cast<double>(w) / h;
  for (const float el : kElevations) {
    SCOPED_TRACE(el);
    const Camera c = makeCamera(w, h, 20.0f, fov, el);
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const double ndcX = 2.0 * (x + 0.5) / w - 1.0;
        const double ndcY = 1.0 - 2.0 * (y + 0.5) / h;
        const Vec3 d = primaryRayDir(c, x, y);
        const double f = dot3(d, c.forward);
        ASSERT_GT(f, 0.0);
        EXPECT_NEAR(dot3(d, c.right) / f, ndcX * aspect * t, 1e-5)
            << "px=" << x << " py=" << y;
        EXPECT_NEAR(dot3(d, c.up) / f, ndcY * t, 1e-5)
            << "px=" << x << " py=" << y;
      }
    }
  }
}

TEST(PrimaryRayTest, LeftRightAndTopBottomPixelsMirror) {
  // Odd and even sizes: pixel (x, y) and its mirror image must have opposite
  // components along `right` (resp. `up`) and identical components along the
  // other two camera axes. Mirror error: ndc is computed from different float
  // numerators, so allow ~2e-7 * a few -> 1e-6.
  struct Size {
    int w, h;
  };
  for (const Size s : {Size{7, 5}, Size{8, 6}, Size{31, 17}}) {
    for (const float el : {0.0f, 30.0f, 60.0f}) {
      SCOPED_TRACE(::testing::Message() << s.w << "x" << s.h << " el=" << el);
      const Camera c = makeCamera(s.w, s.h, 20.0f, 60.0f, el);
      for (int y = 0; y < s.h; ++y) {
        for (int x = 0; x < s.w; ++x) {
          const Vec3 d = primaryRayDir(c, x, y);
          ASSERT_NEAR(len3(d), 1.0, 1e-6);  // guards against 0 == 0

          const Vec3 m = primaryRayDir(c, s.w - 1 - x, y);  // left-right
          EXPECT_NEAR(dot3(d, c.right), -dot3(m, c.right), 1e-6);
          EXPECT_NEAR(dot3(d, c.up), dot3(m, c.up), 1e-6);
          EXPECT_NEAR(dot3(d, c.forward), dot3(m, c.forward), 1e-6);

          const Vec3 v = primaryRayDir(c, x, s.h - 1 - y);  // top-bottom
          EXPECT_NEAR(dot3(d, c.up), -dot3(v, c.up), 1e-6);
          EXPECT_NEAR(dot3(d, c.right), dot3(v, c.right), 1e-6);
          EXPECT_NEAR(dot3(d, c.forward), dot3(v, c.forward), 1e-6);
        }
      }
    }
  }
}

TEST(PrimaryRayTest, TopRowCenterAngleMatchesAnalyticVerticalExtent) {
  // Odd width puts a pixel column exactly on the axis, so the ray through the
  // top row's center lies in the forward/up plane at angle
  //   atan(tanHalfFovY * (1 - 1/H))
  // from forward (pixel CENTER, not the image edge: ndcY = 1 - 1/H). Tall
  // image (H > W) so a swapped aspect would also show up. atan/tan of floats:
  // 1e-5 as specified.
  for (const float fov : {60.0f, 90.0f}) {
    for (const float el : {0.0f, 30.0f}) {
      const int w = 9;
      const int h = 33;
      SCOPED_TRACE(::testing::Message() << "fov=" << fov << " el=" << el);
      const Camera c = makeCamera(w, h, 20.0f, fov, el);
      const double t = std::tan(0.5 * fov * kDeg);
      const double expected = std::atan(t * (1.0 - 1.0 / h));

      const Vec3 top = primaryRayDir(c, (w - 1) / 2, 0);
      EXPECT_NEAR(angleBetween(top, c.forward), expected, 1e-5);
      // In the forward/up plane, tilted toward +up.
      EXPECT_GT(dot3(top, c.up), 0.0);
      EXPECT_NEAR(dot3(top, c.right), 0.0, 1e-6);

      const Vec3 bottom = primaryRayDir(c, (w - 1) / 2, h - 1);
      EXPECT_NEAR(angleBetween(bottom, c.forward), expected, 1e-5);
      EXPECT_LT(dot3(bottom, c.up), 0.0);
    }
  }
}

TEST(PrimaryRayTest, LeftColumnCenterAngleUsesAspectRatio) {
  // Odd height puts a pixel row on the axis. Wide image 16x9 (aspect 16/9):
  // ray through the left column's center is at
  //   atan(aspect * tanHalfFovY * (1 - 1/W))
  // from forward, in the forward/right plane. A swapped (H/W) or missing
  // aspect gives a very different angle.
  const int w = 16;
  const int h = 9;
  for (const float fov : {60.0f, 40.0f}) {
    SCOPED_TRACE(fov);
    const Camera c = makeCamera(w, h, 20.0f, fov, 0.0f);
    const double t = std::tan(0.5 * fov * kDeg);
    const double aspect = static_cast<double>(w) / h;
    const double expected = std::atan(aspect * t * (1.0 - 1.0 / w));

    const Vec3 left = primaryRayDir(c, 0, (h - 1) / 2);
    EXPECT_NEAR(angleBetween(left, c.forward), expected, 1e-5);
    EXPECT_LT(dot3(left, c.right), 0.0);
    EXPECT_NEAR(dot3(left, c.up), 0.0, 1e-6);

    const Vec3 right = primaryRayDir(c, w - 1, (h - 1) / 2);
    EXPECT_NEAR(angleBetween(right, c.forward), expected, 1e-5);
    EXPECT_GT(dot3(right, c.right), 0.0);
  }
}

TEST(PrimaryRayTest, CornerAngleMatchesAnalyticAndGrowsWithFov) {
  // Corner pixel (0,0), un-normalized ray = forward + a*right + b*up with
  // a = -aspect*t*(1-1/W), b = t*(1-1/H): angle = atan(sqrt(a^2 + b^2)).
  const int w = 20;
  const int h = 12;
  double previous = 0.0;
  for (const float fov : {20.0f, 45.0f, 60.0f, 90.0f, 120.0f, 150.0f}) {
    SCOPED_TRACE(fov);
    const Camera c = makeCamera(w, h, 20.0f, fov, 15.0f);
    const double t = std::tan(0.5 * fov * kDeg);
    const double a = static_cast<double>(w) / h * t * (1.0 - 1.0 / w);
    const double b = t * (1.0 - 1.0 / h);
    const double expected = std::atan(std::sqrt(a * a + b * b));

    const double angle = angleBetween(primaryRayDir(c, 0, 0), c.forward);
    EXPECT_NEAR(angle, expected, 1e-5);
    EXPECT_GT(angle, previous);  // strictly larger fov -> larger corner angle
    previous = angle;
  }
}

TEST(PrimaryRayTest, SubPixelAtCentersEqualsIntegerPixelRay) {
  for (const float el : {0.0f, 30.0f}) {
    const Camera c = makeCamera(11, 6, 20.0f, 75.0f, el);
    for (int y = 0; y < c.height; ++y) {
      for (int x = 0; x < c.width; ++x) {
        const Vec3 a = primaryRayDir(c, x, y);
        const Vec3 b = primaryRayDirSub(c, x + 0.5f, y + 0.5f);
        ASSERT_NEAR(len3(a), 1.0, 1e-6);  // guards against 0 == 0
        // x + 0.5f is exact in float, so both paths see identical inputs; the
        // slack only covers a different-but-equivalent evaluation order.
        EXPECT_NEAR(a.x, b.x, 1e-6) << "px=" << x << " py=" << y;
        EXPECT_NEAR(a.y, b.y, 1e-6) << "px=" << x << " py=" << y;
        EXPECT_NEAR(a.z, b.z, 1e-6) << "px=" << x << " py=" << y;
      }
    }
  }
}

TEST(PrimaryRayTest, SubPixelRaysFollowContinuousPixelCoordinates) {
  // Linear in continuous pixel coordinates: fx = 0 is the left image edge,
  // fx = W the right edge, fy = 0 the top edge, fy = H the bottom edge.
  // Same projective oracle as above, at arbitrary fractional positions.
  const int w = 40;
  const int h = 30;
  const float fov = 50.0f;
  const double t = std::tan(0.5 * fov * kDeg);
  const double aspect = static_cast<double>(w) / h;
  const std::vector<std::pair<float, float>> samples = {
      {0.0f, 0.0f},  {40.0f, 30.0f}, {0.0f, 30.0f},
      {40.0f, 0.0f}, {20.0f, 15.0f}, {3.25f, 7.75f},
      {19.9f, 0.1f}, {37.5f, 29.5f}, {10.125f, 22.5f}};
  for (const float el : {0.0f, 40.0f}) {
    const Camera c = makeCamera(w, h, 20.0f, fov, el);
    for (const auto& [fx, fy] : samples) {
      SCOPED_TRACE(::testing::Message()
                   << "fx=" << fx << " fy=" << fy << " el=" << el);
      const double ndcX = 2.0 * fx / w - 1.0;
      const double ndcY = 1.0 - 2.0 * fy / h;
      const Vec3 d = primaryRayDirSub(c, fx, fy);
      EXPECT_NEAR(len3(d), 1.0, 1e-6);
      const double f = dot3(d, c.forward);
      ASSERT_GT(f, 0.0);
      EXPECT_NEAR(dot3(d, c.right) / f, ndcX * aspect * t, 1e-5);
      EXPECT_NEAR(dot3(d, c.up) / f, ndcY * t, 1e-5);
    }
  }
}

TEST(PrimaryRayTest, SubPixelHalfPixelOffsetsStraddleTheCenterRay) {
  // Moving half a pixel left/right of a center gives rays whose `right`
  // tangent differs from the center's by exactly +/- (aspect*t)/W.
  const int w = 20;
  const int h = 10;
  const Camera c = makeCamera(w, h, 20.0f, 60.0f, 20.0f);
  const double step = c.aspect * c.tanHalfFovY / w;  // tangent per half pixel
  const Vec3 mid = primaryRayDirSub(c, 7.5f, 4.5f);
  const Vec3 lo = primaryRayDirSub(c, 7.0f, 4.5f);
  const Vec3 hi = primaryRayDirSub(c, 8.0f, 4.5f);
  const double tm = dot3(mid, c.right) / dot3(mid, c.forward);
  const double tl = dot3(lo, c.right) / dot3(lo, c.forward);
  const double th = dot3(hi, c.right) / dot3(hi, c.forward);
  EXPECT_NEAR(th - tm, step, 1e-5);
  EXPECT_NEAR(tm - tl, step, 1e-5);
}

// --- skyGradient
// --------------------------------------------------------------

TEST(SkyGradientTest, AxisDirectionsMapToDocumentedColors) {
  struct Case {
    Vec3 dir;
    float r, g, b;
  };
  const Case cases[] = {
      {{1, 0, 0}, 1.0f, 0.5f, 0.5f}, {{-1, 0, 0}, 0.0f, 0.5f, 0.5f},
      {{0, 1, 0}, 0.5f, 1.0f, 0.5f}, {{0, -1, 0}, 0.5f, 0.0f, 0.5f},
      {{0, 0, 1}, 0.5f, 0.5f, 1.0f}, {{0, 0, -1}, 0.5f, 0.5f, 0.0f},
  };
  for (const Case& k : cases) {
    SCOPED_TRACE(::testing::Message()
                 << k.dir.x << "," << k.dir.y << "," << k.dir.z);
    const Vec3 col = skyGradient(k.dir);
    EXPECT_FLOAT_EQ(col.x, k.r);
    EXPECT_FLOAT_EQ(col.y, k.g);
    EXPECT_FLOAT_EQ(col.z, k.b);
  }
}

TEST(SkyGradientTest, IsAffineInDirectionAndStaysInUnitRange) {
  // (0.5 + 0.5 * dir): check an arbitrary unit vector against the formula and
  // that all primary-ray colors of a wide-fov camera are within [0, 1].
  const Vec3 d{0.48f, -0.6f, 0.64f};  // unit: .2304 + .36 + .4096 = 1
  const Vec3 col = skyGradient(d);
  EXPECT_NEAR(col.x, 0.74f, 1e-6);
  EXPECT_NEAR(col.y, 0.2f, 1e-6);
  EXPECT_NEAR(col.z, 0.82f, 1e-6);

  const Camera c = makeCamera(32, 24, 20.0f, 150.0f, 45.0f);
  bool sawNonTrivial = false;
  for (int y = 0; y < c.height; ++y) {
    for (int x = 0; x < c.width; ++x) {
      const Vec3 dir = primaryRayDir(c, x, y);
      const Vec3 s = skyGradient(dir);
      // 1e-6 slack: |dir| is unit only to ~1e-7, components may overshoot 1.
      EXPECT_GE(s.x, -1e-6f);
      EXPECT_LE(s.x, 1.0f + 1e-6f);
      EXPECT_GE(s.y, -1e-6f);
      EXPECT_LE(s.y, 1.0f + 1e-6f);
      EXPECT_GE(s.z, -1e-6f);
      EXPECT_LE(s.z, 1.0f + 1e-6f);
      EXPECT_NEAR(s.x, 0.5 + 0.5 * dir.x, 1e-6);
      EXPECT_NEAR(s.y, 0.5 + 0.5 * dir.y, 1e-6);
      EXPECT_NEAR(s.z, 0.5 + 0.5 * dir.z, 1e-6);
      if (s.x < 0.3f || s.x > 0.7f) sawNonTrivial = true;
    }
  }
  EXPECT_TRUE(sawNonTrivial);  // the wide fov really exercises the gradient
}
