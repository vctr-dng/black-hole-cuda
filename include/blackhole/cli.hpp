#pragma once

#include <string>
#include <vector>

#include "blackhole/render.hpp"

namespace blackhole {

struct CliOptions {
  RenderConfig config;
  std::string outputPath = "blackhole.png";
  bool showHelp = false;
};

/// Parse command-line arguments (excluding the program name).
/// Throws std::invalid_argument with a user-readable message on bad input.
CliOptions parseArgs(const std::vector<std::string>& args);

std::string usage();

}  // namespace blackhole
