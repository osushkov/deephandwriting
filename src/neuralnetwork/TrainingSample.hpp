#pragma once

#include "../common/Common.hpp"
#include "../common/Math.hpp"

struct TrainingSample {
  Vector input;
  Vector expectedOutput;

  TrainingSample() = default;
  TrainingSample(const Vector &input, const Vector &expectedOutput)
      : input(input), expectedOutput(expectedOutput) {}
};

std::ostream &operator<<(std::ostream &stream, const TrainingSample &ts);
