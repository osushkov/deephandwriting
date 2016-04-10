#pragma once

#include "Trainer.hpp"
#include "neuralnetwork/TrainingProvider.hpp"
#include <random>

class DynamicTrainer : public Trainer {
public:
  DynamicTrainer(float startLearnRate, float epsilonRate, float maxLearnRate, float momentumAmount,
                 unsigned startNumSamples, unsigned maxNumSamples, bool useMomentum,
                 bool useSpeedup, bool useWeightRates);

  virtual ~DynamicTrainer();

  void Train(Network &network, vector<TrainingSample> &trainingSamples,
             unsigned iterations) override;

  void AddProgressCallback(NetworkTrainerCallback callback) override;

private:
  struct DynamicTrainerImpl;
  uptr<DynamicTrainerImpl> impl;
};
