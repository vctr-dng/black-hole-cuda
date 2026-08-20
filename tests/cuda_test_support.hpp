#pragma once

#include <cuda_runtime.h>
#include <gtest/gtest.h>

// Skip (rather than fail) CUDA tests on machines without a usable GPU.
#define SKIP_IF_NO_CUDA_DEVICE()                                      \
  do {                                                                \
    int blackhole_device_count = 0;                                   \
    if (cudaGetDeviceCount(&blackhole_device_count) != cudaSuccess || \
        blackhole_device_count == 0) {                                \
      GTEST_SKIP() << "no CUDA device available";                     \
    }                                                                 \
  } while (0)
