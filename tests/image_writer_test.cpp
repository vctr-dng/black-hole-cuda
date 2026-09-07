#include <gtest/gtest.h>

#include "blackhole/image.hpp"
#include "blackhole/render.hpp"

// stb_image implementation is provided by src/sky.cpp (linked via blackhole).
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <vector>

#include "stb_image.h"

namespace fs = std::filesystem;

namespace {

// Unique per test so parallel ctest runs don't collide.
fs::path tempPng() {
  const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
  return fs::temp_directory_path() /
         (std::string("blackhole_") + info->test_suite_name() + "_" +
          info->name() + ".png");
}

struct ScopedFile {
  explicit ScopedFile(fs::path p) : path(std::move(p)) {}
  ~ScopedFile() {
    std::error_code ec;
    fs::remove(path, ec);
  }
  fs::path path;
};

}  // namespace

TEST(ImageWriterTest, WritesPngFile) {
  const int width = 8;
  const int height = 8;
  const blackhole::Image image =
      blackhole::generateSolidColor(width, height, {255, 0, 0, 255});
  ScopedFile file(tempPng());

  ASSERT_TRUE(blackhole::write_png(file.path.string(), image));
  EXPECT_TRUE(fs::exists(file.path));
}

TEST(ImageWriterTest, FileIsNotEmpty) {
  const int width = 8;
  const int height = 8;
  const blackhole::Image image =
      blackhole::generateSolidColor(width, height, {255, 0, 0, 255});
  ScopedFile file(tempPng());

  ASSERT_TRUE(blackhole::write_png(file.path.string(), image));
  ASSERT_TRUE(fs::exists(file.path));
  EXPECT_GT(fs::file_size(file.path), 0u);
}

TEST(ImageWriterTest, ProducesPngSignature) {
  const int width = 8;
  const int height = 8;
  const blackhole::Image image =
      blackhole::generateSolidColor(width, height, {255, 0, 0, 255});
  ScopedFile file(tempPng());
  ASSERT_TRUE(blackhole::write_png(file.path.string(), image));

  std::ifstream in(file.path, std::ios::binary);
  std::array<unsigned char, 8> signature{};
  in.read(reinterpret_cast<char*>(signature.data()), signature.size());
  ASSERT_EQ(in.gcount(), 8);

  const std::array<unsigned char, 8> expected{0x89, 0x50, 0x4E, 0x47,
                                              0x0D, 0x0A, 0x1A, 0x0A};
  EXPECT_EQ(signature, expected);
}

TEST(ImageWriterTest, RoundTripsPixelsLosslessly) {
  // Non-square, distinct x/y values: catches stride and transpose bugs.
  const int width = 16;
  const int height = 9;
  const blackhole::Image image =
      blackhole::generateCoordinateImage(width, height);
  ScopedFile file(tempPng());
  ASSERT_TRUE(blackhole::write_png(file.path.string(), image));

  int w = 0, h = 0, channels = 0;
  unsigned char* decoded =
      stbi_load(file.path.string().c_str(), &w, &h, &channels, 4);
  ASSERT_NE(decoded, nullptr) << stbi_failure_reason();
  const std::vector<unsigned char> bytes(
      decoded,
      decoded + static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 4);
  stbi_image_free(decoded);

  ASSERT_EQ(w, image.width());
  ASSERT_EQ(h, image.height());
  ASSERT_EQ(bytes.size(), image.size());
  EXPECT_TRUE(std::equal(bytes.begin(), bytes.end(), image.data()));
}

TEST(ImageWriterTest, RejectsBadArgs) {
  const blackhole::Image image(2, 2);
  EXPECT_FALSE(blackhole::write_png("", image));
  EXPECT_FALSE(blackhole::write_png("/nonexistent-dir/x.png", image));

  std::vector<std::uint8_t> rgba(16, 255);
  EXPECT_FALSE(blackhole::write_png("", 2, 2, rgba.data()));
  EXPECT_FALSE(blackhole::write_png("/tmp/x.png", 0, 2, rgba.data()));
  EXPECT_FALSE(blackhole::write_png("/tmp/x.png", 2, 2, nullptr));
}
