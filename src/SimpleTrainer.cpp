
#include "SimpleTrainer.hpp"
#include <cassert>

SimpleTrainer::SimpleTrainer(float startLearnRate, float endLearnRate, unsigned stochasticSamples)
    : startLearnRate(startLearnRate), endLearnRate(endLearnRate),
      stochasticSamples(stochasticSamples) {

  assert(startLearnRate > endLearnRate);
  assert(endLearnRate >= 0.0f);
  assert(stochasticSamples > 0);
}

void SimpleTrainer::Train(Network &network, vector<TrainingSample> &trainingSamples,
                          unsigned iterations) {

  random_shuffle(trainingSamples.begin(), trainingSamples.end());
  curSamplesIndex = 0;

  for (unsigned i = 0; i < iterations; i++) {
    float lr = getLearnRate(i, iterations);

    TrainingProvider samplesProvider = getStochasticSamples(trainingSamples);
    pair<Tensor, float> gradientError = network.ComputeGradient(samplesProvider);
    network.ApplyUpdate(gradientError.first * -lr);

    for_each(trainingCallbacks, [&network, &gradientError, i](const NetworkTrainerCallback &cb) {
      cb(network, gradientError.second, i);
    });
  }
}

void SimpleTrainer::AddProgressCallback(NetworkTrainerCallback callback) {
  trainingCallbacks.push_back(callback);
}

float SimpleTrainer::getLearnRate(unsigned curIter, unsigned iterations) {
  return startLearnRate + (endLearnRate - startLearnRate) * curIter / (float)iterations;
}

TrainingProvider SimpleTrainer::getStochasticSamples(vector<TrainingSample> &allSamples) {
  unsigned numSamples = min<unsigned>(allSamples.size(), stochasticSamples);

  if ((curSamplesIndex + numSamples) >= allSamples.size()) {
    random_shuffle(allSamples.begin(), allSamples.end());
    curSamplesIndex = 0;
  }

  auto result = TrainingProvider(allSamples, numSamples, curSamplesIndex);
  curSamplesIndex += numSamples;

  return result;
}
