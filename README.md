# Black Hole Cuda

CUDA renderer for a static Schwarzschild black hole: one GPU thread per pixel traces a light ray through the curved spacetime.

## Requirements

- gcc-14
- CMake >= 3.24
- Ninja
- CUDA/nvcc 13
- sm_120 by default (RTX 50-series); override with `-DCMAKE_CUDA_ARCHITECTURES=...`

## Getting started

Build :

```bash
cmake --preset default && cmake --build build
```

Render an image :

```bash
./build/blackhole_cli --width 1920 --height 1080 --elevation 10 \
    --sky starfield --disk --post --spp 4 --bloom 0.3 -o out/final.png
```

> [!TIP]
> `./build/blackhole_cli --help` provides a complete list of flags.

Run tests :

```bash
ctest --test-dir build
```


## How it works

Units: Schwarzschild radius rs = 1, c = 1. Light rays follow the exact photon "force" $a = -1.5 rs h^2 x / r^5$ ($h = |x \times v|$ conserved), integrated with adaptive velocity Verlet (`include/blackhole/geodesic.hpp`).

| Piece | Header | Notes |
|---|---|---|
| Geodesic integrator | `geodesic.hpp` | Verlet + RK4, capture / escape tracing |
| Camera | `camera.hpp` | pinhole, elevation above the disk plane |
| Sky | `sky.hpp`, `src/sky.cpp` | procedural starfield, equirectangular image via stb_image + CUDA texture |
| Disk | `disk.hpp` | $\text{T} \sim r^{-3/4}$, blackbody colour LUT, Doppler + gravitational g-factor, intensity $\sim g^3$, plane-crossing test |
| Post | `post.hpp` | ACES tone map, exact sRGB, stratified jittered supersampling, Gaussian bloom |
| Renderer | `src/render.cu`, `render.hpp` | kernels, config validation |

Disk model: opaque, thin, first crossing of the plane y = 0 inside [diskInner, diskOuter] wins.

## TODO

- Optimization :
  - Precomputed deflection table for the static case
- An interactive viewer
- Spinning (Kerr) black holes
