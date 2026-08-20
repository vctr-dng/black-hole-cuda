cmake_minimum_required(VERSION 3.24)

project(black-hole-cuda
  VERSION 0.1.0
  DESCRIPTION "CUDA black hole renderer"
  LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# RTX 5080 (Blackwell) is sm_120; override with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

include(FetchContent)
include(CTest)

# --- stb (stb_image_write) ----------------------------------------------------
FetchContent_Declare(
  stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG        2c980bb59875b0d32144a71867fbdebb2f77cd20
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(stb)

add_library(stb_image_write INTERFACE)
add_library(stb::image_write ALIAS stb_image_write)
target_include_directories(stb_image_write INTERFACE
  ${stb_SOURCE_DIR}
)

# --- GoogleTest ---------------------------------------------------------------
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(googletest)

# --- Library ------------------------------------------------------------------
add_library(blackhole
  src/render.cu
  src/image.cpp
  src/image_buffer.cpp
  src/cli.cpp
)
add_library(blackhole::blackhole ALIAS blackhole)

target_include_directories(blackhole
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(blackhole
  PUBLIC
    stb::image_write
)

set_target_properties(blackhole PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

target_compile_options(blackhole PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# --- Executable ---------------------------------------------------------------
add_executable(blackhole_cli src/main.cpp)
target_link_libraries(blackhole_cli PRIVATE blackhole::blackhole)

# --- Tests --------------------------------------------------------------------
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
