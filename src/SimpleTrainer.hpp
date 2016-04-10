#pragma once

#include "Trainer.hpp"
#include "neuralnetwork/TrainingProvider.hpp"

#include <vector>

class SimpleTrainer : public Trainer {
public:
  SimpleTrainer(float startLearnRate, float endLearnRate, unsigned stochasticSamples);
  virtual ~SimpleTrainer() = default;

  void Train(Network &network, vector<TrainingSample> &trainingSamples,
             unsigned iterations) override;

  void AddProgressCallback(NetworkTrainerCallback callback) override;

private:
  const float startLearnRate;
  const float endLearnRate;
  const unsigned stochasticSamples;

  unsigned curSamplesIndex;
  vector<NetworkTrainerCallback> trainingCallbacks;

  float getLearnRate(unsigned curIter, unsigned iterations);
  TrainingProvider getStochasticSamples(vector<TrainingSample> &allSamples);
};
