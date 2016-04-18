
#include "AdamTrainer.hpp"
#include <cassert>
#include <iostream>
#include <random>

struct AdamTrainer::AdamTrainerImpl {
  static constexpr float beta1 = 0.9f;
  static constexpr float beta2 = 0.999f;
  static constexpr float epsilon = 10e-8;
  static constexpr float lr = 0.001f;

  mt19937 rnd;
  vector<NetworkTrainerCallback> trainingCallbacks;

  unsigned numCompletePasses;
  unsigned curSamplesIndex;
  unsigned curSamplesOffset;

  AdamTrainerImpl() {
    random_device rd;
    this->rnd = mt19937(rd());
  }

  void Train(Network &network, vector<TrainingSample> &trainingSamples, unsigned iterations,
             GradientRestriction *restriction) {
    shuffle(trainingSamples.begin(), trainingSamples.end(), rnd);
    numCompletePasses = 0;

    curSamplesIndex = 0;
    curSamplesOffset = 0;

    Tensor momentum = initialMomentum(network);
    Tensor rms = initialRMS(network);

    for (unsigned i = 0; i < iterations; i++) {
      TrainingProvider samplesProvider = getStochasticSamples(trainingSamples, i, iterations);

      pair<Tensor, float> gradientError = network.ComputeGradient(samplesProvider);
      if (restriction != nullptr) {
        restriction->RestrictGradient(gradientError.first);
      }

      updateMomentum(momentum, gradientError.first);
      updateRMS(rms, gradientError.first);

      Tensor weightUpdate = computeWeightUpdate(momentum, rms);
      network.ApplyUpdate(weightUpdate);

      for_each(trainingCallbacks, [&network, &gradientError, i](const NetworkTrainerCallback &cb) {
        cb(network, gradientError.second, i);
      });
    }
  }

  void AddProgressCallback(NetworkTrainerCallback callback) {
    trainingCallbacks.push_back(callback);
  }

  Tensor computeWeightUpdate(const Tensor &momentum, const Tensor &rms) {
    Tensor update = momentum; // just to get the dimensionality right.

    for (unsigned i = 0; i < rms.NumLayers(); i++) {
      for (int y = 0; y < rms(i).rows(); y++) {
        for (int x = 0; x < rms(i).cols(); x++) {
          float mc = momentum(i)(y, x) / (1.0f - beta1);
          float rc = rms(i)(y, x) / (1.0f - beta2);

          update(i)(y, x) = -lr * mc / sqrtf(rc + epsilon);
        }
      }
    }

    return update;
  }

  Tensor initialMomentum(Network &network) {
    Tensor result = network.Weights();
    for (unsigned i = 0; i < result.NumLayers(); i++) {
      result(i).fill(0.0f);
    }
    return result;
  }

  Tensor initialRMS(Network &network) {
    Tensor result = network.Weights();
    for (unsigned i = 0; i < result.NumLayers(); i++) {
      result(i).fill(0.0f);
    }
    return result;
  }

  void updateMomentum(Tensor &momentum, const Tensor &gradient) {
    assert(gradient.NumLayers() == momentum.NumLayers());
    momentum = momentum * beta1 + gradient * (1.0f - beta1);
  }

  void updateRMS(Tensor &rms, const Tensor &gradient) {
    assert(gradient.NumLayers() == rms.NumLayers());

    for (unsigned i = 0; i < gradient.NumLayers(); i++) {
      assert(gradient(i).rows() == rms(i).rows());
      assert(gradient(i).cols() == rms(i).cols());

      for (int y = 0; y < gradient(i).rows(); y++) {
        for (int x = 0; x < gradient(i).cols(); x++) {
          rms(i)(y, x) =
              beta2 * rms(i)(y, x) + (1.0f - beta2) * gradient(i)(y, x) * gradient(i)(y, x);
        }
      }
    }
  }

  TrainingProvider getStochasticSamples(vector<TrainingSample> &allSamples, unsigned curIter,
                                        unsigned totalIters) {
    unsigned numSamples =
        min<unsigned>(allSamples.size(), numStochasticSamples(curIter, totalIters));

    if ((curSamplesIndex + numSamples) > allSamples.size()) {
      if (numCompletePasses % 10 == 0) {
        shuffle(allSamples.begin(), allSamples.end(), rnd);
      } else {
        curSamplesOffset = rnd() % allSamples.size();
      }
      curSamplesIndex = 0;
      numCompletePasses++;
    }

    auto result = TrainingProvider(allSamples, numSamples, curSamplesIndex + curSamplesOffset);
    curSamplesIndex += numSamples;

    return result;
  }

  unsigned numStochasticSamples(unsigned curIter, unsigned totalIters) {
    assert(curIter <= totalIters);
    return 2000;
  }
};

AdamTrainer::AdamTrainer() : impl(make_unique<AdamTrainerImpl>()) {}
AdamTrainer::~AdamTrainer() = default;

void AdamTrainer::Train(Network &network, vector<TrainingSample> &trainingSamples,
                        unsigned iterations, GradientRestriction *restriction) {
  impl->Train(network, trainingSamples, iterations, restriction);
}

void AdamTrainer::AddProgressCallback(NetworkTrainerCallback callback) {
  impl->AddProgressCallback(callback);
}
