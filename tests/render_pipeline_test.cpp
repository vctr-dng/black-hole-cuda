#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>

#include "blackhole/render.hpp"
#include "cuda_test_support.hpp"

// CUDA -> device buffer -> host Image -> stb PNG writer -> file on disk.
TEST(RenderPipelineTest, GeneratesPng) {
  SKIP_IF_NO_CUDA_DEVICE();

  const int width = 16;
  const int height = 9;
  const blackhole::Image image =
      blackhole::renderTestPatternCuda(width, height);
  ASSERT_EQ(image.width(), width);
  ASSERT_EQ(image.height(), height);

  const auto path =
      std::filesystem::temp_directory_path() / "blackhole_pipeline_test.png";
  ASSERT_TRUE(blackhole::write_png(path.string(), image));
  ASSERT_TRUE(std::filesystem::exists(path));
  EXPECT_GT(std::filesystem::file_size(path), 50u);

  std::ifstream in(path, std::ios::binary);
  std::array<unsigned char, 8> sig{};
  in.read(reinterpret_cast<char*>(sig.data()), sig.size());
  const std::array<unsigned char, 8> expected{0x89, 0x50, 0x4E, 0x47,
                                              0x0D, 0x0A, 0x1A, 0x0A};
  EXPECT_EQ(sig, expected);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}
