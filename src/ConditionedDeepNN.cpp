
#include "ConditionedDeepNN.hpp"

#include "DataLoader.hpp"
#include "ExperimentUtils.hpp"
#include "common/Common.hpp"
#include "learningpolicies/AdamTrainer.hpp"
#include "learningpolicies/DynamicTrainer.hpp"
#include "learningpolicies/DynamicTrainerBuilder.hpp"
#include "learningpolicies/SimpleTrainer.hpp"
#include "neuralnetwork/Autoencoder.hpp"
#include "neuralnetwork/Network.hpp"
#include "util/Timer.hpp"
#include "util/Util.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using Eigen::MatrixXd;

static Network createNewNetwork(unsigned inputSize, unsigned outputSize);

static vector<TrainingSample>
autoencodedSamplesFromTrainingData(const vector<TrainingSample> &samples);

static vector<TrainingSample>
autoencodedSamplesFromNetworkLayer(Network &network, unsigned layer,
                                   const vector<TrainingSample> &samples);

static void conditionNetwork(Network &network, vector<TrainingSample> &trainingSamples);

static uptr<Trainer> getTrainer(void);

static void learn(Network &network, vector<TrainingSample> &trainingSamples,
                  vector<TrainingSample> &testSamples);

void ConditionedDeepNN::TrainAndEvaluate(void) {
  cout << "loading training data" << endl;
  vector<TrainingSample> trainingSamples =
      DataLoader::loadSamples("data/train_images.idx3", "data/train_labels.idx1", true);
  random_shuffle(trainingSamples.begin(), trainingSamples.end());
  cout << "training data size: " << trainingSamples.size() << endl;

  cout << "loading test data" << endl;
  vector<TrainingSample> testSamples =
      DataLoader::loadSamples("data/test_images.idx3", "data/test_labels.idx1", false);
  random_shuffle(testSamples.begin(), testSamples.end());
  cout << "test data size: " << testSamples.size() << endl;

  unsigned inputSize = trainingSamples.front().input.rows();
  unsigned outputSize = trainingSamples.front().expectedOutput.rows();

  Network network = createNewNetwork(inputSize, outputSize);
  learn(network, trainingSamples, testSamples);
}

Network createNewNetwork(unsigned inputSize, unsigned outputSize) {
  auto hiddenActivation = make_shared<Tanh>();

  NetworkSpec spec;
  spec.numInputs = inputSize;
  spec.numOutputs = outputSize;
  spec.outputFunc = make_shared<Logistic>();
  // spec.hiddenLayers = {make_pair(inputSize / 4, hiddenActivation)};
  spec.hiddenLayers = {make_pair(inputSize, hiddenActivation),
                       make_pair(inputSize / 2, hiddenActivation)};
  spec.nodeActivationRate = 0.6f;
  spec.softmaxOutput = true;

  return Network(spec);
}

vector<TrainingSample> autoencodedSamplesFromTrainingData(const vector<TrainingSample> &samples) {
  vector<TrainingSample> result;
  result.reserve(samples.size());
  for (const auto &ts : samples) {
    result.push_back(TrainingSample(ts.input, ts.input));
  }
  return result;
}

vector<TrainingSample> autoencodedSamplesFromNetworkLayer(Network &network, unsigned layer,
                                                          const vector<TrainingSample> &samples) {
  vector<TrainingSample> result;
  result.reserve(samples.size());
  for (const auto &ts : samples) {
    Vector v = network.Process(ts.input, layer);
    result.push_back(TrainingSample(v, v));
  }
  return result;
}

void conditionNetwork(Network &network, vector<TrainingSample> &trainingSamples) {
  cout << "conditioning" << endl;

  for (unsigned layer = 0; layer < network.NumLayers() - 1; layer++) {
    cout << "conditioning layer: " << layer << endl;
    vector<TrainingSample> conditionSubsample =
        layer == 0 ? autoencodedSamplesFromTrainingData(trainingSamples)
                   : autoencodedSamplesFromNetworkLayer(network, layer, trainingSamples);

    auto fitFunc = network.LayerActivationFunc(layer);
    auto outputFunc = make_unique<Linear>();

    Autoencoder autoEncoder(0.25f);
    Matrix hl = autoEncoder.ComputeHiddenLayer(network.LayerSize(layer), move(fitFunc),
                                               conditionSubsample, move(outputFunc));
    network.SetLayerWeights(layer, hl);
  }

  cout << "finished conditioning" << endl;
}

uptr<Trainer> getTrainer(void) {
  return uptr<Trainer>(new AdamTrainer());
  // DynamicTrainerBuilder builder;
  //
  // builder.StartLearnRate(0.01f)
  //     .FinishLearnRate(0.001f)
  //     .MaxLearnRate(0.1f)
  //     .Momentum(0.25f)
  //     .StartSamplesPerIter(5000)
  //     .FinishSamplesPerIter(5000)
  //     .UseMomentum(true)
  //     .UseSpeedup(true)
  //     .UseWeightRates(true);
  //
  // return builder.Build();
}

void learn(Network &network, vector<TrainingSample> &trainingSamples,
           vector<TrainingSample> &testSamples) {
  conditionNetwork(network, trainingSamples);

  auto trainer = getTrainer();
  trainer->AddProgressCallback(
      [&trainingSamples, &testSamples](Network &network, float trainError, unsigned iter) {
        if (iter % 100 == 0) {
          float testWrong = ExperimentUtils::eval(network, testSamples);
          cout << iter << "\t" << testWrong << " " << trainError << endl;
          // cout << iter << "\t" << trainError << "\t" << testWrong << endl;

          // float trainWrong = testNetwork(network, trainingSamples);
          // cout << iter << "\t" << trainError << "\t" << testWrong << "\t" << trainWrong << endl;
        }
      });

  cout << "starting training..." << endl;

  // AllowSelectLayers restrictLayers({1});
  trainer->Train(network, trainingSamples, 30000, nullptr);
  cout << "finished" << endl;
}
