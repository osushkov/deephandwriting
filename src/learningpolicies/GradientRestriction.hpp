#pragma once

#include "../common/Common.hpp"
#include "../neuralnetwork/Tensor.hpp"
#include <set>

class GradientRestriction {
public:
  virtual ~GradientRestriction() = default;

  virtual void RestrictGradient(Tensor &gradient) = 0;
};

class AllowSelectLayers : public GradientRestriction {
  set<unsigned> trainLayers; // which layers will be trained.
public:
  AllowSelectLayers(set<unsigned> trainLayers) : trainLayers(trainLayers) {}

  void RestrictGradient(Tensor &gradient) {
    for (unsigned i = 0; i < gradient.NumLayers(); i++) {
      if (trainLayers.find(i) == trainLayers.end()) {
        gradient(i).fill(0.0f);
      }
    }
  }
};
