#include "blackhole/cli.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

using blackhole::CliOptions;
using blackhole::parseArgs;

TEST(CliTest, DefaultsWhenNoArguments) {
  const CliOptions opts = parseArgs({});

  EXPECT_EQ(opts.config.width, 800);
  EXPECT_EQ(opts.config.height, 600);
  EXPECT_FLOAT_EQ(opts.config.observerDistance, 20.0f);
  EXPECT_FLOAT_EQ(opts.config.fovDegrees, 60.0f);
  EXPECT_EQ(opts.outputPath, "blackhole.png");
  EXPECT_FALSE(opts.showHelp);
}

TEST(CliTest, ParsesAllOptions) {
  const CliOptions opts =
      parseArgs({"--width", "320", "--height", "200", "--distance", "12.5",
                 "--fov", "90", "--output", "out.png"});

  EXPECT_EQ(opts.config.width, 320);
  EXPECT_EQ(opts.config.height, 200);
  EXPECT_FLOAT_EQ(opts.config.observerDistance, 12.5f);
  EXPECT_FLOAT_EQ(opts.config.fovDegrees, 90.0f);
  EXPECT_EQ(opts.outputPath, "out.png");
}

TEST(CliTest, ParsesShortOutputAndEqualsForm) {
  EXPECT_EQ(parseArgs({"-o", "a.png"}).outputPath, "a.png");
  EXPECT_EQ(parseArgs({"--width=64"}).config.width, 64);
}

TEST(CliTest, PositionalArgumentIsOutputPath) {
  EXPECT_EQ(parseArgs({"image.png"}).outputPath, "image.png");
}

TEST(CliTest, HelpFlag) {
  EXPECT_TRUE(parseArgs({"--help"}).showHelp);
  EXPECT_TRUE(parseArgs({"-h"}).showHelp);
  EXPECT_FALSE(blackhole::usage().empty());
}

TEST(CliTest, RejectsUnknownOption) {
  EXPECT_THROW(parseArgs({"--bogus"}), std::invalid_argument);
}

TEST(CliTest, RejectsMissingValue) {
  EXPECT_THROW(parseArgs({"--width"}), std::invalid_argument);
}

TEST(CliTest, RejectsNonNumericValues) {
  EXPECT_THROW(parseArgs({"--width", "abc"}), std::invalid_argument);
  EXPECT_THROW(parseArgs({"--width", "12abc"}), std::invalid_argument);
  EXPECT_THROW(parseArgs({"--fov", "wide"}), std::invalid_argument);
}

TEST(CliTest, RejectsInvalidConfigValues) {
  EXPECT_THROW(parseArgs({"--width", "0"}), std::invalid_argument);
  EXPECT_THROW(parseArgs({"--height", "-5"}), std::invalid_argument);
  EXPECT_THROW(parseArgs({"--distance", "0"}), std::invalid_argument);
  EXPECT_THROW(parseArgs({"--fov", "0"}), std::invalid_argument);
  EXPECT_THROW(parseArgs({"--fov", "180"}), std::invalid_argument);
}

TEST(CliTest, RejectsSecondPositionalArgument) {
  EXPECT_THROW(parseArgs({"a.png", "b.png"}), std::invalid_argument);
}
