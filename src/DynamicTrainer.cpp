
#include "DynamicTrainer.hpp"
#include <cassert>
#include <iostream>
#include <random>

struct DynamicTrainer::DynamicTrainerImpl {
  const float startLearnRate;
  const float epsilonRate;
  const float maxLearnRate;
  const float momentumAmount;
  const unsigned startNumSamples;
  const unsigned maxNumSamples;

  const bool useMomentum;
  const bool useSpeedup;
  const bool useWeightRates;

  mt19937 rnd;

  unsigned numCompletePasses;
  unsigned curSamplesIndex;
  unsigned curSamplesOffset;
  float curLearnRate;

  vector<NetworkTrainerCallback> trainingCallbacks;

  DynamicTrainerImpl(float startLearnRate, float epsilonRate, float maxLearnRate,
                     float momentumAmount, unsigned startNumSamples, unsigned maxNumSamples,
                     bool useMomentum, bool useSpeedup, bool useWeightRates)
      : startLearnRate(startLearnRate), epsilonRate(epsilonRate), maxLearnRate(maxLearnRate),
        momentumAmount(momentumAmount), startNumSamples(startNumSamples),
        maxNumSamples(maxNumSamples), useMomentum(useMomentum), useSpeedup(useSpeedup),
        useWeightRates(useWeightRates) {

    assert(startLearnRate > 0.0f);
    assert(epsilonRate > 0.0f && epsilonRate < startLearnRate);
    assert(maxLearnRate > 0.0f);
    assert(momentumAmount >= 0.0f && momentumAmount < 1.0f);
    assert(startNumSamples > 0);
    assert(maxNumSamples >= startNumSamples);

    random_device rd;
    this->rnd = mt19937(rd());
  }

  void Train(Network &network, vector<TrainingSample> &trainingSamples, unsigned iterations,
             GradientRestriction *restriction) {
    shuffle(trainingSamples.begin(), trainingSamples.end(), rnd);
    numCompletePasses = 0;
    curSamplesIndex = 0;
    curSamplesOffset = 0;

    curLearnRate = startLearnRate;
    Tensor momentum = initialMomentum(network);
    Tensor rmsWeights = initialRMS(network);

    Tensor weightGradientRate;
    Tensor prevGradient;
    float prevError = 0.0f;

    for (unsigned i = 0; i < iterations; i++) {
      TrainingProvider samplesProvider = getStochasticSamples(trainingSamples, i, iterations);

      pair<Tensor, float> gradientError =
          updateMomentum(momentum, network, samplesProvider, restriction);

      if (i == 0) {
        prevGradient = gradientError.first;
        prevError = gradientError.second;

        weightGradientRate = gradientError.first;
        initWeightGradientRates(weightGradientRate);
      } else {
        if (useWeightRates) {
          updateWeightsGradientRates(prevGradient, gradientError.first, weightGradientRate);
        }
        prevGradient = gradientError.first;
      }

      updateGradientRMS(gradientError.first, rmsWeights);
      network.ApplyUpdate(momentum * weightGradientRate * rmsScaling(rmsWeights));

      curLearnRate *= pow(epsilonRate / startLearnRate, 1.0f / iterations);
      prevError = gradientError.second;

      for_each(trainingCallbacks, [&network, &gradientError, i](const NetworkTrainerCallback &cb) {
        cb(network, gradientError.second, i);
      });
    }
  }

  pair<Tensor, float> updateMomentum(Tensor &momentum, Network &network, TrainingProvider &samples,
                                     GradientRestriction *restriction) {
    Network cpy(network);
    cpy.ApplyUpdate(momentum * momentumAmount);

    pair<Tensor, float> gradientError = cpy.ComputeGradient(samples);
    if (restriction != nullptr) {
      restriction->RestrictGradient(gradientError.first);
    }

    momentum = momentum * momentumAmount + gradientError.first * -curLearnRate;
    return gradientError;
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
      result(i).fill(1.0f);
    }
    return result;
  }

  void AddProgressCallback(NetworkTrainerCallback callback) {
    trainingCallbacks.push_back(callback);
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

  void initWeightGradientRates(Tensor &rates) {
    for (unsigned i = 0; i < rates.NumLayers(); i++) {
      rates(i).fill(1.0f);
    }
  }

  void updateWeightsGradientRates(const Tensor &prevGradient, const Tensor &curGradient,
                                  Tensor &rates) {

    assert(curGradient.NumLayers() == prevGradient.NumLayers());
    assert(curGradient.NumLayers() == rates.NumLayers());

    for (unsigned i = 0; i < curGradient.NumLayers(); i++) {
      assert(curGradient(i).rows() == prevGradient(i).rows());
      assert(curGradient(i).cols() == prevGradient(i).cols());
      assert(curGradient(i).rows() == rates(i).rows());
      assert(curGradient(i).cols() == rates(i).cols());

      for (int y = 0; y < curGradient(i).rows(); y++) {
        for (int x = 0; x < curGradient(i).cols(); x++) {
          float prev = prevGradient(i)(y, x);
          float cur = curGradient(i)(y, x);

          if ((prev >= 0.0f && cur >= 0.0f) || (prev <= 0.0f && cur <= 0.0f)) {
            rates(i)(y, x) += 0.05f;
          } else {
            rates(i)(y, x) *= 0.95f;
          }

          rates(i)(y, x) = min(rates(i)(y, x), 10.0f);
          rates(i)(y, x) = max(rates(i)(y, x), 0.5f);
        }
      }
    }
  }

  void updateGradientRMS(const Tensor &gradient, Tensor &rms) {
    assert(gradient.NumLayers() == rms.NumLayers());

    for (unsigned i = 0; i < gradient.NumLayers(); i++) {
      assert(gradient(i).rows() == rms(i).rows());
      assert(gradient(i).cols() == rms(i).cols());

      for (int y = 0; y < gradient(i).rows(); y++) {
        for (int x = 0; x < gradient(i).cols(); x++) {
          rms(i)(y, x) = 0.9f * rms(i)(y, x) + 0.1f * gradient(i)(y, x) * gradient(i)(y, x);
        }
      }
    }
  }

  Tensor rmsScaling(const Tensor &rms) {
    Tensor result = rms;
    for (unsigned i = 0; i < result.NumLayers(); i++) {
      for (int y = 0; y < result(i).rows(); y++) {
        for (int x = 0; x < result(i).cols(); x++) {
          result(i)(y, x) = 1.0f / sqrtf(rms(i)(y, x) + 0.001);
          // cout << result(i)(y, x) << " ";
        }
      }
    }
    // cout << endl;
    return result;
  }

  unsigned numStochasticSamples(unsigned curIter, unsigned totalIters) {
    assert(curIter <= totalIters);

    float iterFrac = curIter / static_cast<float>(totalIters);
    return startNumSamples + static_cast<unsigned>((maxNumSamples - startNumSamples) * iterFrac);
  }
};

DynamicTrainer::DynamicTrainer(float startLearnRate, float epsilonRate, float maxLearnRate,
                               float momentumAmount, unsigned startNumSamples,
                               unsigned maxNumSamples, bool useMomentum, bool useSpeedup,
                               bool useWeightRates)
    : impl(make_unique<DynamicTrainerImpl>(startLearnRate, epsilonRate, maxLearnRate,
                                           momentumAmount, startNumSamples, maxNumSamples,
                                           useMomentum, useSpeedup, useWeightRates)) {}

DynamicTrainer::~DynamicTrainer() = default;

void DynamicTrainer::Train(Network &network, vector<TrainingSample> &trainingSamples,
                           unsigned iterations, GradientRestriction *restriction) {
  impl->Train(network, trainingSamples, iterations, restriction);
}

void DynamicTrainer::AddProgressCallback(NetworkTrainerCallback callback) {
  impl->AddProgressCallback(callback);
}
