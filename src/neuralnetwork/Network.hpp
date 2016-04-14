#pragma once

#include "../common/Common.hpp"
#include "../common/Math.hpp"
#include "ActivationFunc.hpp"
#include "Tensor.hpp"
#include "TrainingProvider.hpp"
#include <vector>

struct NetworkSpec {
  unsigned numInputs;
  unsigned numOutputs;

  sptr<ActivationFunc> outputFunc;
  vector<pair<unsigned, sptr<ActivationFunc>>> hiddenLayers;
};

class Network {
public:
  static void OutputDebugging(void);

  Network(const Network &other); // copy ctor
  Network(Network &&other); // move ctor

  Network(const NetworkSpec &spec);
  Network(istream &stream); // TODO: this constructor doesnt handle different activation funcs.

  virtual ~Network();

  Vector Process(const Vector &input);
  Vector Process(const Vector &input, unsigned limitLayers);

  pair<Tensor, float> ComputeGradient(const TrainingProvider &samplesProvider);
  void ApplyUpdate(const Tensor &weightUpdates);

  unsigned NumLayers(void) const;
  unsigned LayerSize(unsigned layer) const;

  Tensor Weights(void) const;
  Matrix LayerWeights(unsigned layer) const;
  void SetLayerWeights(unsigned layer, const Matrix &weights);

  std::ostream &Output(ostream &stream);

  void Serialize(ostream &stream) const;

private:
  struct NetworkImpl;
  uptr<NetworkImpl> impl;
};
