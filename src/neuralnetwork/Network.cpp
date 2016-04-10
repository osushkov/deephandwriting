
#include "Network.hpp"
#include "../common/ThreadPool.hpp"
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

static const float INIT_WEIGHT_RANGE = 0.1f;

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
  Tensor zeroGradient;

  NetworkImpl(const vector<unsigned> &layerSizes)
      : numInputs(layerSizes[0]), numOutputs(layerSizes[layerSizes.size() - 1]),
        numLayers(layerSizes.size() - 1) {

    assert(layerSizes.size() >= 2);

    for (unsigned i = 0; i < numLayers; i++) {
      layerWeights.AddLayer(createLayer(layerSizes[i], layerSizes[i + 1]));
    }

    zeroGradient = layerWeights;
    for (unsigned i = 0; i < zeroGradient.NumLayers(); i++) {
      zeroGradient(i).setZero();
    }
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

private:
  Matrix createLayer(unsigned inputSize, unsigned layerSize) {
    assert(inputSize > 0 && layerSize > 0);

    unsigned numRows = layerSize;
    unsigned numCols = inputSize + 1; // +1 accounts for bias input

    Matrix result(numRows, numCols);

    for (unsigned r = 0; r < result.rows(); r++) {
      for (unsigned c = 0; c < result.cols(); c++) {
        result(r, c) = Util::RandInterval(-INIT_WEIGHT_RANGE, INIT_WEIGHT_RANGE);
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
    assert(input.rows() == numInputs);

    ctx.layerOutputs.resize(layerWeights.NumLayers());
    ctx.layerDerivatives.resize(layerWeights.NumLayers());

    auto out = getLayerOutput(input, layerWeights(0), false);
    ctx.layerOutputs[0] = out.first;
    ctx.layerDerivatives[0] = out.second;

    for (unsigned i = 1; i < layerWeights.NumLayers(); i++) {
      auto out = getLayerOutput(ctx.layerOutputs[i - 1], layerWeights(i),
                                i == layerWeights.NumLayers() - 1);
      ctx.layerOutputs[i] = out.first;
      ctx.layerDerivatives[i] = out.second;
    }

    assert(ctx.layerOutputs[ctx.layerOutputs.size() - 1].rows() == numOutputs);
    return ctx.layerOutputs[ctx.layerOutputs.size() - 1];
  }

  // Returns the output vector of the layer, and the derivative vector for the layer.
  pair<Vector, Vector> getLayerOutput(const Vector &prevLayer, const Matrix &layerWeights,
                                      bool isOutput) const {
    Vector z =
        layerWeights.topRightCorner(layerWeights.rows(), layerWeights.cols() - 1) * prevLayer;
    Vector derivatives(z.rows());

    for (unsigned i = 0; i < layerWeights.rows(); i++) {
      z(i) += layerWeights(i, 0);
      float in = z(i);
      z(i) = isOutput ? outputActivationFunc(in) : activationFunc(in);

      derivatives(i) = isOutput ? outputDerivativeFunc(in, z(i)) : derivativeFunc(in, z(i));
    }

    return make_pair(z, derivatives);
  }

  float outputActivationFunc(float v) const { return 1.0f / (1.0f + expf(-v)); }
  // float activationFunc(float v) const { return 1.0f / (1.0f + expf(-v)); } // logistic function
  // float activationFunc(float v) const { return logf(1.0f + expf(v)); } // softplus function
  float activationFunc(float v) const { return v > 0.0f ? v : (0.01f * v); }

  float outputDerivativeFunc(float in, float out) const { return out * (1.0f - out); }
  // float derivativeFunc(float in, float out) const { return out * (1.0f - out); }
  // float derivativeFunc(float in, float out) const { return 1.0f / (1.0f + expf(-in)); }
  float derivativeFunc(float in, float out) const { return in > 0.0f ? 1.0f : 0.01f; }

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
    result(0) = 1.0f;
    result.bottomRightCorner(noBiasInput.rows(), 1) = noBiasInput;
    return result;
  }
};

Network::Network(Network &&other) : impl(move(other.impl)) {}
Network::Network(const vector<unsigned> &layerSizes) : impl(new NetworkImpl(layerSizes)) {}
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

pair<Tensor, float> Network::ComputeGradient(const TrainingProvider &samplesProvider) {
  return impl->ComputeGradient(samplesProvider);
}

void Network::ApplyUpdate(const Tensor &weightUpdates) { impl->ApplyUpdate(weightUpdates); }

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
