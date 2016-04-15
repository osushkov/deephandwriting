#pragma once

#include "../neuralnetwork/TrainingProvider.hpp"
#include "GradientRestriction.hpp"
#include "Trainer.hpp"
#include <random>

class DynamicTrainer : public Trainer {
public:
  DynamicTrainer(float startLearnRate, float epsilonRate, float maxLearnRate, float momentumAmount,
                 unsigned startNumSamples, unsigned maxNumSamples, bool useMomentum,
                 bool useSpeedup, bool useWeightRates);

  virtual ~DynamicTrainer();

  void Train(Network &network, vector<TrainingSample> &trainingSamples, unsigned iterations,
             GradientRestriction *restriction) override;

  void AddProgressCallback(NetworkTrainerCallback callback) override;

private:
  struct DynamicTrainerImpl;
  uptr<DynamicTrainerImpl> impl;
};
