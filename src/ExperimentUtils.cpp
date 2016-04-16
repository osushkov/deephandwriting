
#include "ExperimentUtils.hpp"

static int digitFromNNOutput(const Vector &out);

float ExperimentUtils::eval(Network &network, const vector<TrainingSample> &testSamples) {
  unsigned numCorrect = 0;

  for (const auto &ts : testSamples) {
    auto result = network.Process(ts.input);
    bool isCorrect = digitFromNNOutput(result) == digitFromNNOutput(ts.expectedOutput);
    numCorrect += isCorrect ? 1 : 0;
  }

  return 1.0f - numCorrect / static_cast<float>(testSamples.size());
}

int digitFromNNOutput(const Vector &out) {
  assert(out.rows() == 10);

  int result = 0;
  float maxActivation = out(0);

  for (int i = 1; i < out.rows(); i++) {
    if (out(i) > maxActivation) {
      maxActivation = out(i);
      result = i;
    }
  }

  return result;
}
