
#pragma once

#include "../common/Common.hpp"
#include "../common/Math.hpp"
#include "TrainingProvider.hpp"

class Autoencoder {
public:
  Autoencoder(unsigned inputSize, unsigned hiddenSize);
  virtual ~Autoencoder();

  Matrix ComputeHiddenLayer(vector<TrainingSample> &samples);

private:
  struct AutoencoderImpl;
  uptr<AutoencoderImpl> impl;
};
