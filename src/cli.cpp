#include "blackhole/cli.hpp"

#include <cstddef>
#include <stdexcept>

namespace blackhole {
namespace {

int parseInt(const std::string &option, const std::string &text) {
  std::size_t used = 0;
  int value = 0;
  try {
    value = std::stoi(text, &used);
  } catch (const std::exception &) {
    used = 0;
  }
  if (used == 0 || used != text.size()) {
    throw std::invalid_argument(option + ": expected an integer, got '" + text +
                                "'");
  }
  return value;
}

float parseFloat(const std::string &option, const std::string &text) {
  std::size_t used = 0;
  float value = 0.0f;
  try {
    value = std::stof(text, &used);
  } catch (const std::exception &) {
    used = 0;
  }
  if (used == 0 || used != text.size()) {
    throw std::invalid_argument(option + ": expected a number, got '" + text +
                                "'");
  }
  return value;
}

} // namespace

CliOptions parseArgs(const std::vector<std::string> &args) {
  CliOptions opts;
  bool haveOutput = false;

  for (std::size_t i = 0; i < args.size(); ++i) {
    std::string arg = args[i];
    std::string inlineValue;
    bool hasInline = false;

    const auto eq = arg.find('=');
    if (arg.rfind("--", 0) == 0 && eq != std::string::npos) {
      inlineValue = arg.substr(eq + 1);
      arg = arg.substr(0, eq);
      hasInline = true;
    }

    auto value = [&]() -> std::string {
      if (hasInline)
        return inlineValue;
      if (i + 1 >= args.size()) {
        throw std::invalid_argument(arg + ": missing value");
      }
      return args[++i];
    };

    auto noValue = [&]() {
      if (hasInline) {
        throw std::invalid_argument(arg + ": does not take a value");
      }
    };

    if (arg == "-h" || arg == "--help") {
      opts.showHelp = true;
    } else if (arg == "--width") {
      opts.config.width = parseInt(arg, value());
    } else if (arg == "--height") {
      opts.config.height = parseInt(arg, value());
    } else if (arg == "--distance") {
      opts.config.observerDistance = parseFloat(arg, value());
    } else if (arg == "--fov") {
      opts.config.fovDegrees = parseFloat(arg, value());
    } else if (arg == "--elevation") {
      opts.config.elevationDegrees = parseFloat(arg, value());
    } else if (arg == "--sky") {
      const std::string v = value();
      if (v.empty()) {
        throw std::invalid_argument(arg +
                                    ": expected gradient, starfield or a path");
      }
      if (v == "gradient") {
        opts.config.sky = SkyMode::Gradient;
      } else if (v == "starfield") {
        opts.config.sky = SkyMode::Starfield;
      } else {
        opts.config.sky = SkyMode::Image;
        opts.config.skyImagePath = v;
      }
    } else if (arg == "--disk") {
      noValue();
      opts.config.disk = true;
    } else if (arg == "--disk-inner") {
      opts.config.diskInner = parseFloat(arg, value());
    } else if (arg == "--disk-outer") {
      opts.config.diskOuter = parseFloat(arg, value());
    } else if (arg == "--disk-temp") {
      opts.config.diskTempK = parseFloat(arg, value());
    } else if (arg == "--disk-brightness") {
      opts.config.diskBrightness = parseFloat(arg, value());
    } else if (arg == "--post") {
      noValue();
      opts.config.postProcess = true;
    } else if (arg == "--exposure") {
      opts.config.exposure = parseFloat(arg, value());
    } else if (arg == "--spp") {
      opts.config.spp = parseInt(arg, value());
    } else if (arg == "--bloom") {
      opts.config.bloomStrength = parseFloat(arg, value());
    } else if (arg == "-o" || arg == "--output") {
      opts.outputPath = value();
      haveOutput = true;
    } else if (!arg.empty() && arg[0] == '-') {
      throw std::invalid_argument("unknown option: " + arg);
    } else {
      if (haveOutput) {
        throw std::invalid_argument("unexpected extra argument: " + arg);
      }
      opts.outputPath = arg;
      haveOutput = true;
    }
  }

  validate(opts.config);
  return opts;
}

std::string usage() {
  return "usage: blackhole_cli [options] [output.png]\n"
         "  --width N       image width in pixels (default 800)\n"
         "  --height N      image height in pixels (default 600)\n"
         "  --distance D    observer distance from the black hole, finite and "
         "> 1 (default 20)\n"
         "  --fov DEG       field of view in degrees, 0 < DEG < 180 (default "
         "60)\n"
         "  --elevation DEG camera elevation above the disk plane, 0 <= DEG < "
         "90 (default 10)\n"
         "  --sky S         sky: gradient, starfield, or an image path "
         "(default gradient)\n"
         "  --disk          enable the accretion disk\n"
         "  --disk-inner R  disk inner radius, R > 1 (default 3)\n"
         "  --disk-outer R  disk outer radius, R > inner (default 12)\n"
         "  --disk-temp K   disk temperature in Kelvin, 1000..40000 (default "
         "12000)\n"
         "  --disk-brightness X  disk brightness, X >= 0 (default 1)\n"
         "  --post          enable post-processing\n"
         "  --exposure X    exposure multiplier, X > 0 (default 1)\n"
         "  --spp N         samples per pixel: 1, 4, 9, 16, 25, 36, 49 or 64 "
         "(default 1)\n"
         "  --bloom X       bloom strength, X >= 0 (default 0)\n"
         "  -o, --output F  output PNG path (default blackhole.png)\n"
         "  -h, --help      show this message\n";
}

} // namespace blackhole
