
#pragma once

#include "../common/Common.hpp"
#include "../common/Math.hpp"
#include "ActivationFunc.hpp"
#include "TrainingProvider.hpp"

class Autoencoder {
public:
  Autoencoder(float pLoss);
  virtual ~Autoencoder();

  Matrix ComputeHiddenLayer(unsigned layerSize, uptr<ActivationFunc> hiddenFunc,
                            const vector<TrainingSample> &samples,
                            uptr<ActivationFunc> dataModelingFunc);

private:
  struct AutoencoderImpl;
  uptr<AutoencoderImpl> impl;
};
