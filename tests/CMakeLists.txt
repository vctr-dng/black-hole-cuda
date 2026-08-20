include(GoogleTest)

add_executable(blackhole_tests
  image_test.cpp
  image_generator_test.cpp
  image_writer_test.cpp
  cli_test.cpp
  cuda_image_test.cpp
  render_pipeline_test.cpp
  render_test.cpp
)

target_link_libraries(blackhole_tests
  PRIVATE
    blackhole::blackhole
    CUDA::cudart
    GTest::gtest_main
)

gtest_discover_tests(blackhole_tests)
