
#pragma once

#include "../neuralnetwork/TrainingProvider.hpp"
#include "GradientRestriction.hpp"
#include "Trainer.hpp"
#include <random>

class AdamTrainer : public Trainer {
public:
  AdamTrainer();
  virtual ~AdamTrainer();

  void Train(Network &network, vector<TrainingSample> &trainingSamples, unsigned iterations,
             GradientRestriction *restriction) override;

  void AddProgressCallback(NetworkTrainerCallback callback) override;

private:
  struct AdamTrainerImpl;
  uptr<AdamTrainerImpl> impl;
};
