
#include "Network.hpp"
#include "../util/Timer.hpp"
#include "../util/Util.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/tbb.h>

#include <atomic>
#include <cassert>
#include <cmath>
#include <future>
#include <iostream>
#include <thread>

struct NetworkContext {
  vector<Vector> layerOutputs;
  vector<Vector> layerDerivatives;
  vector<Vector> layerDeltas;
};

struct Network::NetworkImpl {
  const unsigned numInputs;
  const unsigned numOutputs;
  const unsigned numLayers;

  Tensor layerWeights;
  vector<sptr<ActivationFunc>> layerActivations;

  Tensor zeroGradient;

  NetworkImpl(const NetworkSpec &spec)
      : numInputs(spec.numInputs), numOutputs(spec.numOutputs),
        numLayers(spec.hiddenLayers.size() + 1) {

    if (spec.hiddenLayers.empty()) {
      layerWeights.AddLayer(createLayer(numInputs, numOutputs));
      layerActivations.push_back(spec.outputFunc);
    } else {
      layerWeights.AddLayer(createLayer(numInputs, spec.hiddenLayers[0].first));
      layerActivations.push_back(spec.hiddenLayers[0].second);

      for (unsigned i = 0; i < spec.hiddenLayers.size() - 1; i++) {
        layerWeights.AddLayer(
            createLayer(spec.hiddenLayers[i].first, spec.hiddenLayers[i + 1].first));
        layerActivations.push_back(spec.hiddenLayers[i + 1].second);
      }

      layerWeights.AddLayer(
          createLayer(spec.hiddenLayers[spec.hiddenLayers.size() - 1].first, numOutputs));
      layerActivations.push_back(spec.outputFunc);
    }

    zeroGradient = layerWeights;
    for (unsigned i = 0; i < zeroGradient.NumLayers(); i++) {
      zeroGradient(i).setZero();
    }
  }

  NetworkImpl(const NetworkImpl &other)
      : numInputs(other.numInputs), numOutputs(other.numOutputs),
        numLayers(other.numLayers) {
    layerWeights = other.layerWeights;
    layerActivations = other.layerActivations;
    zeroGradient = other.zeroGradient;
  }

  NetworkImpl(unsigned numInputs, unsigned numOutputs, unsigned numLayers,
              const Tensor &layerWeights)
      : numInputs(numInputs), numOutputs(numOutputs), numLayers(numLayers),
        layerWeights(layerWeights) {

    assert(numInputs > 0 && numLayers >= 2 && numOutputs > 0);

    zeroGradient = layerWeights;
    for (unsigned i = 0; i < zeroGradient.NumLayers(); i++) {
      zeroGradient(i).setZero();
    }
  }

  Vector Process(const Vector &input) {
    assert(input.rows() == numInputs);

    NetworkContext ctx;
    return process(input, ctx);
  }

  Vector Process(const Vector &input, unsigned limitLayers) {
    assert(input.rows() == numInputs);

    NetworkContext ctx;
    return process(input, limitLayers, ctx);
  }

  pair<Tensor, float> ComputeGradient(const TrainingProvider &samplesProvider) {
    const unsigned numSubsets = tbb::task_scheduler_init::default_num_threads();

    auto gradient = make_pair(zeroGradient, 0.0f);
    Tensor &netGradient = gradient.first;
    float &error = gradient.second;

    mutex gradientMutex;

    auto gradientWorker = [this, numSubsets, &samplesProvider, &netGradient, &error,
                           &gradientMutex](const tbb::blocked_range<unsigned> &r) {
      for (unsigned i = r.begin(); i != r.end(); i++) {
        unsigned start = (i * samplesProvider.NumSamples()) / numSubsets;
        unsigned end = ((i + 1) * samplesProvider.NumSamples()) / numSubsets;

        auto subsetGradient = computeGradientSubset(samplesProvider, start, end);

        {
          std::unique_lock<std::mutex> lock(gradientMutex);
          netGradient += subsetGradient.first;
          error += subsetGradient.second;
        }
      }
    };

    tbb::parallel_for(tbb::blocked_range<unsigned>(0, numSubsets), gradientWorker);

    float scaleFactor = 1.0f / samplesProvider.NumSamples();
    netGradient *= scaleFactor;
    error *= scaleFactor;

    return gradient;
  }

  void ApplyUpdate(const Tensor &weightUpdates) { layerWeights += weightUpdates; }

  Matrix LayerWeights(unsigned layer) const {
    assert(layer < layerWeights.NumLayers());
    return layerWeights(layer);
  }

  void SetLayerWeights(unsigned layer, const Matrix &weights) {
    assert(layer < layerWeights.NumLayers());
    assert(layerWeights(layer).rows() == weights.rows());
    assert(layerWeights(layer).cols() == weights.cols());

    layerWeights(layer) = weights;
  }

private:
  Matrix createLayer(unsigned inputSize, unsigned layerSize) {
    assert(inputSize > 0 && layerSize > 0);

    float ic = 0.1f;

    unsigned numRows = layerSize;
    unsigned numCols = inputSize + 1; // +1 accounts for bias input
    float initRange = 1.0f / sqrtf(numCols * ic);

    Matrix result(numRows, numCols);
    result.fill(0.0f);

    for (unsigned r = 0; r < result.rows(); r++) {
      for (unsigned c = 0; c < static_cast<unsigned>(result.cols() * ic); c++) {
        unsigned col = rand() % result.cols();
        result(r, col) = Util::RandInterval(-initRange, initRange);
      }
    }

    return result;
  }

  pair<Tensor, float> computeGradientSubset(const TrainingProvider &samplesProvider, unsigned start,
                                            unsigned end) const {

    auto gradient = make_pair(zeroGradient, 0.0f);
    NetworkContext ctx;

    for (unsigned i = start; i < end; i++) {
      computeSampleGradient(samplesProvider.GetSample(i), ctx, gradient);
    }

    return gradient;
  }

  Vector process(const Vector &input, NetworkContext &ctx) const {
    return process(input, layerWeights.NumLayers(), ctx);
  }

  Vector process(const Vector &input, unsigned limitLayers, NetworkContext &ctx) const {
    assert(input.rows() == numInputs);

    ctx.layerOutputs.resize(layerWeights.NumLayers());
    ctx.layerDerivatives.resize(layerWeights.NumLayers());

    auto out = getLayerOutput(input, layerWeights(0), layerActivations[0].get());
    ctx.layerOutputs[0] = out.first;
    ctx.layerDerivatives[0] = out.second;

    for (unsigned i = 1; i < limitLayers; i++) {
      out = getLayerOutput(ctx.layerOutputs[i - 1], layerWeights(i), layerActivations[i].get());
      ctx.layerOutputs[i] = out.first;
      ctx.layerDerivatives[i] = out.second;
    }

    return out.first;
  }

  // Returns the output vector of the layer, and the derivative vector for the layer.
  pair<Vector, Vector> getLayerOutput(const Vector &prevLayer, const Matrix &layerWeights,
                                      ActivationFunc *func) const {
    Vector z =
        layerWeights.topRightCorner(layerWeights.rows(), layerWeights.cols() - 1) * prevLayer;
    Vector derivatives(z.rows());

    for (unsigned i = 0; i < layerWeights.rows(); i++) {
      z(i) += layerWeights(i, 0);
      float in = z(i);

      z(i) = func->ActivationValue(in);
      derivatives(i) = func->DerivativeValue(in, z(i));
    }

    return make_pair(z, derivatives);
  }

  void computeSampleGradient(const TrainingSample &sample, NetworkContext &ctx,
                             pair<Tensor, float> &outGradient) const {
    Vector output = process(sample.input, ctx);

    ctx.layerDeltas.resize(numLayers);
    ctx.layerDeltas[ctx.layerDeltas.size() - 1] =
        output - sample.expectedOutput; // cross entropy error function.

    for (int i = ctx.layerDeltas.size() - 2; i >= 0; i--) {
      Matrix noBiasWeights = layerWeights(i + 1).bottomRightCorner(layerWeights(i + 1).rows(),
                                                                   layerWeights(i + 1).cols() - 1);
      ctx.layerDeltas[i] = noBiasWeights.transpose() * ctx.layerDeltas[i + 1];

      assert(ctx.layerDeltas[i].rows() == ctx.layerOutputs[i].rows());
      for (unsigned r = 0; r < ctx.layerDeltas[i].rows(); r++) {
        ctx.layerDeltas[i](r) *= ctx.layerDerivatives[i](r);
      }
    }

    for (unsigned i = 0; i < numLayers; i++) {
      Matrix &og = outGradient.first(i);
      const auto &ld = ctx.layerDeltas[i];
      const auto inputs = getInputWithBias(i == 0 ? sample.input : ctx.layerOutputs[i - 1]);

      for (unsigned r = 0; r < ld.rows(); r++) {
        for (unsigned c = 0; c < inputs.rows(); c++) {
          og(r, c) += inputs(c) * ld(r);
        }
      }
    }

    for (unsigned i = 0; i < output.rows(); i++) {
      outGradient.second +=
          (output(i) - sample.expectedOutput(i)) * (output(i) - sample.expectedOutput(i));
    }
  }

  Vector getInputWithBias(const Vector &noBiasInput) const {
    Vector result(noBiasInput.rows() + 1);
    result(noBiasInput.rows()) = 1.0f;
    result.topRightCorner(noBiasInput.rows(), 1) = noBiasInput;
    return result;
  }
};

Network::Network(const Network &other) : impl(new NetworkImpl(*other.impl)) {}
Network::Network(Network &&other) : impl(move(other.impl)) {}

Network::Network(const NetworkSpec &spec) : impl(new NetworkImpl(spec)) {}

Network::Network(istream &stream) {
  unsigned numInputs, numOutputs, numLayers;

  stream.read((char *)&numInputs, sizeof(unsigned));
  stream.read((char *)&numOutputs, sizeof(unsigned));
  stream.read((char *)&numLayers, sizeof(unsigned));

  Tensor layerWeights;
  layerWeights.Deserialize(stream);

  impl = make_unique<NetworkImpl>(numInputs, numOutputs, numLayers, layerWeights);
}
Network::~Network() = default;

Vector Network::Process(const Vector &input) { return impl->Process(input); }
Vector Network::Process(const Vector &input, unsigned limitLayers) {
  return impl->Process(input, limitLayers);
}

pair<Tensor, float> Network::ComputeGradient(const TrainingProvider &samplesProvider) {
  return impl->ComputeGradient(samplesProvider);
}

void Network::ApplyUpdate(const Tensor &weightUpdates) { impl->ApplyUpdate(weightUpdates); }

unsigned Network::NumLayers(void) const { return impl->layerWeights.NumLayers(); }

unsigned Network::LayerSize(unsigned layer) const { return impl->layerWeights(layer).rows(); }

Tensor Network::Weights(void) const { return impl->layerWeights; }

Matrix Network::LayerWeights(unsigned layer) const { return impl->LayerWeights(layer); }

void Network::SetLayerWeights(unsigned layer, const Matrix &weights) {
  impl->SetLayerWeights(layer, weights);
}

std::ostream &Network::Output(ostream &stream) {
  for (unsigned i = 0; i < impl->layerWeights.NumLayers(); i++) {
    for (unsigned r = 0; r < impl->layerWeights(i).rows(); r++) {
      for (unsigned c = 0; c < impl->layerWeights(i).cols(); c++) {
        stream << impl->layerWeights(i)(r, c) << " ";
      }
      stream << endl;
    }
    stream << endl;
  }
  return stream;
}

void Network::Serialize(std::ostream &stream) const {
  stream.write((char *)&impl->numInputs, sizeof(unsigned));
  stream.write((char *)&impl->numOutputs, sizeof(unsigned));
  stream.write((char *)&impl->numLayers, sizeof(unsigned));

  impl->layerWeights.Serialize(stream);
}
