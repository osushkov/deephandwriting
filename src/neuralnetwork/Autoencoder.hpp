
#pragma once

#include "../common/Common.hpp"
#include "../common/Math.hpp"
#include "ActivationFunc.hpp"
#include "TrainingProvider.hpp"

enum class EncodedDataType {
  BOUNDED_NORMALISED, // between 0 and 1
  UNBOUNDED,          // can be any real number
};

class Autoencoder {
public:
  Autoencoder(float pLoss);
  virtual ~Autoencoder();

  Matrix ComputeHiddenLayer(unsigned layerSize, uptr<ActivationFunc> hiddenFunc,
                            const vector<TrainingSample> &samples, EncodedDataType dataType);

private:
  struct AutoencoderImpl;
  uptr<AutoencoderImpl> impl;
};
