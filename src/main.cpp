#include <iostream>
#include <string>
#include <vector>

#include "blackhole/cli.hpp"
#include "blackhole/render.hpp"

int main(int argc, char** argv) {
  try {
    const blackhole::CliOptions opts =
        blackhole::parseArgs(std::vector<std::string>(argv + 1, argv + argc));
    if (opts.showHelp) {
      std::cout << blackhole::usage();
      return 0;
    }

    const blackhole::Image image = blackhole::render(opts.config);
    if (!blackhole::write_png(opts.outputPath, image)) {
      std::cerr << "failed to write PNG: " << opts.outputPath << '\n';
      return 1;
    }
    std::cout << "wrote " << opts.outputPath << " (" << image.width() << "x"
              << image.height() << ")\n";
  } catch (const std::invalid_argument& ex) {
    std::cerr << "error: " << ex.what() << "\n\n" << blackhole::usage();
    return 2;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << '\n';
    return 1;
  }

  return 0;
}
