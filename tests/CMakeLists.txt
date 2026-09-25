include(GoogleTest)

add_executable(blackhole_tests
  image_test.cpp
  image_generator_test.cpp
  image_writer_test.cpp
  cli_test.cpp
  cuda_image_test.cpp
  render_pipeline_test.cpp
  render_test.cpp
  geodesic_test.cpp
  geodesic_device_test.cu
  camera_test.cpp
  render_flat_test.cpp
  render_shadow_test.cpp
  render_post_test.cpp
  render_sky_test.cpp
  sky_test.cpp
  sky_gpu_test.cu
  post_test.cpp
  post_device_test.cu
  disk_test.cpp
  disk_device_test.cu
  render_disk_test.cpp
)

target_link_libraries(blackhole_tests
  PRIVATE
    blackhole::blackhole
    CUDA::cudart
    GTest::gtest_main
)

# geodesic_device_test.cu is the only CUDA source here. Match the blackhole
# library (relocatable device code, relaxed constexpr) so device linking against
# it works and the shared header's constants are usable from kernels.
set_target_properties(blackhole_tests PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
target_compile_options(blackhole_tests PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

gtest_discover_tests(blackhole_tests)
