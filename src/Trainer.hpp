#pragma once

#include "common/Common.hpp"
#include "neuralnetwork/Network.hpp"

#include <functional>
#include <vector>

// A callback function that is called after every iteration of the trainer. The arguments
// are the network, the current training error, and the iteration number.
using NetworkTrainerCallback = function<void(Network &, float, unsigned)>;

class Trainer {
public:
  virtual ~Trainer() {}

  virtual void Train(Network &network, vector<TrainingSample> &trainingSamples,
                     unsigned iterations) = 0;

  virtual void AddProgressCallback(NetworkTrainerCallback callback) = 0;
};
