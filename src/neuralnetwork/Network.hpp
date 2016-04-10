#pragma once

#include "../common/Common.hpp"
#include "../common/Math.hpp"
#include "Tensor.hpp"
#include "TrainingProvider.hpp"
#include <vector>

class Network {
public:
  static void OutputDebugging(void);

  Network(Network &&other);
  Network(const vector<unsigned> &layerSizes);
  Network(istream &stream);

  virtual ~Network();

  Vector Process(const Vector &input);
  pair<Tensor, float> ComputeGradient(const TrainingProvider &samplesProvider);
  void ApplyUpdate(const Tensor &weightUpdates);

  std::ostream &Output(ostream &stream);

  void Serialize(ostream &stream) const;

private:
  struct NetworkImpl;
  uptr<NetworkImpl> impl;
};
