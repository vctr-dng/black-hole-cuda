#include "blackhole/cli.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <vector>

using blackhole::CliOptions;
using blackhole::parseArgs;

namespace {

// Expects parseArgs to reject `args` with an error that is about the elevation
// value, not the generic "unknown option" a CLI without --elevation would give.
void expectElevationError(const std::vector<std::string> &args) {
  try {
    parseArgs(args);
    ADD_FAILURE() << "expected std::invalid_argument";
  } catch (const std::invalid_argument &e) {
    std::string msg = e.what();
    std::transform(msg.begin(), msg.end(), msg.begin(), [](unsigned char ch) {
      return static_cast<char>(std::tolower(ch));
    });
    EXPECT_NE(msg.find("elevation"), std::string::npos) << msg;
    EXPECT_EQ(msg.find("unknown option"), std::string::npos) << msg;
  }
}

} // namespace

TEST(CliTest, DefaultsWhenNoArguments) {
  const CliOptions opts = parseArgs({});

  EXPECT_EQ(opts.config.width, 800);
  EXPECT_EQ(opts.config.height, 600);
  EXPECT_FLOAT_EQ(opts.config.observerDistance, 20.0f);
  EXPECT_FLOAT_EQ(opts.config.fovDegrees, 60.0f);
  EXPECT_FLOAT_EQ(opts.config.elevationDegrees, 10.0f);
  EXPECT_EQ(opts.outputPath, "blackhole.png");
  EXPECT_FALSE(opts.showHelp);
}

TEST(CliTest, ParsesAllOptions) {
  const CliOptions opts =
      parseArgs({"--width", "320", "--height", "200", "--distance", "12.5",
                 "--fov", "90", "--elevation", "35.5", "--output", "out.png"});

  EXPECT_EQ(opts.config.width, 320);
  EXPECT_EQ(opts.config.height, 200);
  EXPECT_FLOAT_EQ(opts.config.observerDistance, 12.5f);
  EXPECT_FLOAT_EQ(opts.config.fovDegrees, 90.0f);
  EXPECT_FLOAT_EQ(opts.config.elevationDegrees, 35.5f);
  EXPECT_EQ(opts.outputPath, "out.png");
}

TEST(CliTest, ParsesShortOutputAndEqualsForm) {
  EXPECT_EQ(parseArgs({"-o", "a.png"}).outputPath, "a.png");
  EXPECT_EQ(parseArgs({"--width=64"}).config.width, 64);
}

TEST(CliTest, ParsesElevation) {
  EXPECT_FLOAT_EQ(parseArgs({"--elevation", "30"}).config.elevationDegrees,
                  30.0f);
  EXPECT_FLOAT_EQ(parseArgs({"--elevation=45.5"}).config.elevationDegrees,
                  45.5f);
  // Boundaries of the valid range [0, 90).
  EXPECT_FLOAT_EQ(parseArgs({"--elevation", "0"}).config.elevationDegrees,
                  0.0f);
  EXPECT_FLOAT_EQ(parseArgs({"--elevation", "89.9"}).config.elevationDegrees,
                  89.9f);
  // Does not disturb the other options.
  const CliOptions opts = parseArgs({"--elevation", "5", "--fov", "70"});
  EXPECT_FLOAT_EQ(opts.config.elevationDegrees, 5.0f);
  EXPECT_FLOAT_EQ(opts.config.fovDegrees, 70.0f);
}

TEST(CliTest, PositionalArgumentIsOutputPath) {
  EXPECT_EQ(parseArgs({"image.png"}).outputPath, "image.png");
}

TEST(CliTest, HelpFlag) {
  EXPECT_TRUE(parseArgs({"--help"}).showHelp);
  EXPECT_TRUE(parseArgs({"-h"}).showHelp);
  EXPECT_FALSE(blackhole::usage().empty());
  EXPECT_NE(blackhole::usage().find("--elevation"), std::string::npos);
}

TEST(CliTest, RejectsUnknownOption) {
  EXPECT_THROW(parseArgs({"--bogus"}), std::invalid_argument);
}

TEST(CliTest, RejectsMissingValue) {
  EXPECT_THROW(parseArgs({"--width"}), std::invalid_argument);
  expectElevationError({"--elevation"});
}

TEST(CliTest, RejectsNonNumericValues) {
  EXPECT_THROW(parseArgs({"--width", "abc"}), std::invalid_argument);
  EXPECT_THROW(parseArgs({"--width", "12abc"}), std::invalid_argument);
  EXPECT_THROW(parseArgs({"--fov", "wide"}), std::invalid_argument);
  expectElevationError({"--elevation", "up"});
  expectElevationError({"--elevation", "10deg"});
  expectElevationError({"--elevation="});
}

TEST(CliTest, RejectsInvalidConfigValues) {
  EXPECT_THROW(parseArgs({"--width", "0"}), std::invalid_argument);
  EXPECT_THROW(parseArgs({"--height", "-5"}), std::invalid_argument);
  EXPECT_THROW(parseArgs({"--distance", "0"}), std::invalid_argument);
  EXPECT_THROW(parseArgs({"--fov", "0"}), std::invalid_argument);
  EXPECT_THROW(parseArgs({"--fov", "180"}), std::invalid_argument);
}

TEST(CliTest, RejectsOutOfRangeElevation) {
  expectElevationError({"--elevation", "-1"});
  expectElevationError({"--elevation", "90"});
  expectElevationError({"--elevation", "180"});
  expectElevationError({"--elevation", "nan"});
}

TEST(CliTest, RejectsSecondPositionalArgument) {
  EXPECT_THROW(parseArgs({"a.png", "b.png"}), std::invalid_argument);
}

namespace {

std::string lowerMsg(const std::vector<std::string> &args) {
  try {
    parseArgs(args);
  } catch (const std::invalid_argument &e) {
    std::string msg = e.what();
    std::transform(msg.begin(), msg.end(), msg.begin(), [](unsigned char ch) {
      return static_cast<char>(std::tolower(ch));
    });
    return msg;
  }
  ADD_FAILURE() << "expected std::invalid_argument";
  return {};
}

// Expects a rejection that mentions `needle` and is not "unknown option".
void expectErrorMentions(const std::vector<std::string> &args,
                         const std::string &needle) {
  const std::string msg = lowerMsg(args);
  EXPECT_NE(msg.find(needle), std::string::npos) << msg;
  EXPECT_EQ(msg.find("unknown option"), std::string::npos) << msg;
}

} // namespace

TEST(CliTest, WiringDefaultsUnchanged) {
  const CliOptions opts = parseArgs({});
  EXPECT_EQ(opts.config.sky, blackhole::SkyMode::Gradient);
  EXPECT_FALSE(opts.config.disk);
}

TEST(CliTest, ParsesSkyKeywordsAndPaths) {
  EXPECT_EQ(parseArgs({"--sky", "gradient"}).config.sky,
            blackhole::SkyMode::Gradient);
  EXPECT_EQ(parseArgs({"--sky=starfield"}).config.sky,
            blackhole::SkyMode::Starfield);
  CliOptions opts = parseArgs({"--sky", "textures/milkyway.hdr"});
  EXPECT_EQ(opts.config.sky, blackhole::SkyMode::Image);
  EXPECT_EQ(opts.config.skyImagePath, "textures/milkyway.hdr");
  opts = parseArgs({"--sky=/abs/sky.png"});
  EXPECT_EQ(opts.config.sky, blackhole::SkyMode::Image);
  EXPECT_EQ(opts.config.skyImagePath, "/abs/sky.png");
}

TEST(CliTest, RejectsEmptyOrMissingSky) {
  expectErrorMentions({"--sky"}, "--sky");
  expectErrorMentions({"--sky="}, "sky");
  expectErrorMentions({"--sky", ""}, "sky");
}

TEST(CliTest, ParsesBooleanFlags) {
  EXPECT_TRUE(parseArgs({"--disk"}).config.disk);
  // Boolean flags take no value: the next token is a positional output path.
  const CliOptions opts = parseArgs({"--disk", "out.png"});
  EXPECT_TRUE(opts.config.disk);
  EXPECT_EQ(opts.outputPath, "out.png");
  EXPECT_THROW(parseArgs({"--disk=1"}), std::invalid_argument);
}

TEST(CliTest, ParsesNumericWiringOptions) {
  const CliOptions opts =
      parseArgs({"--disk-inner", "2.5", "--disk-outer=20", "--disk-temp",
                 "8000", "--disk-brightness=2"});
  EXPECT_FLOAT_EQ(opts.config.diskInner, 2.5f);
  EXPECT_FLOAT_EQ(opts.config.diskOuter, 20.0f);
  EXPECT_FLOAT_EQ(opts.config.diskTempK, 8000.0f);
  EXPECT_FLOAT_EQ(opts.config.diskBrightness, 2.0f);
}

TEST(CliTest, RejectsMissingWiringValues) {
  expectErrorMentions({"--disk-inner"}, "--disk-inner");
  expectErrorMentions({"--disk-outer"}, "--disk-outer");
  expectErrorMentions({"--disk-temp"}, "--disk-temp");
  expectErrorMentions({"--disk-brightness"}, "--disk-brightness");
}

TEST(CliTest, RejectsNonNumericWiringValues) {
  expectErrorMentions({"--disk-inner", "x"}, "--disk-inner");
  expectErrorMentions({"--disk-outer=big"}, "--disk-outer");
  expectErrorMentions({"--disk-temp", "hot"}, "--disk-temp");
  expectErrorMentions({"--disk-brightness", "1x"}, "--disk-brightness");
}

TEST(CliTest, RejectsOutOfRangeWiringValues) {
  expectErrorMentions({"--disk-inner", "1"}, "diskinner");
  expectErrorMentions({"--disk-inner", "nan"}, "diskinner");
  expectErrorMentions({"--disk-outer", "3"}, "diskouter");
  expectErrorMentions({"--disk-outer", "inf"}, "diskouter");
  expectErrorMentions({"--disk-temp", "999"}, "disktempk");
  expectErrorMentions({"--disk-temp", "40001"}, "disktempk");
  expectErrorMentions({"--disk-brightness", "-1"}, "diskbrightness");
}

TEST(CliTest, UsageMentionsEveryWiringFlag) {
  const std::string text = blackhole::usage();
  for (const char *flag :
       {"--sky", "--disk ", "--disk-inner", "--disk-outer", "--disk-temp",
        "--disk-brightness"}) {
    EXPECT_NE(text.find(flag), std::string::npos) << flag;
  }
}
