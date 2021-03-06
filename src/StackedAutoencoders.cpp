
#include "StackedAutoencoders.hpp"

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

static Network createFeatureExtractor(unsigned inputSize);
static Network createClassifier(unsigned inputSize, unsigned outputSize);

static vector<TrainingSample>
autoencodedSamplesFromTrainingData(const vector<TrainingSample> &samples);

static vector<TrainingSample>
autoencodedSamplesFromNetworkLayer(Network &network, unsigned layer,
                                   const vector<TrainingSample> &samples);

static void trainFeatureExtractor(Network &network, vector<TrainingSample> &trainingSamples);

static vector<TrainingSample> featureExtract(Network &network, vector<TrainingSample> &samples);

static void trainClassifier(Network &network, vector<TrainingSample> &trainingSamples,
                            vector<TrainingSample> &testSamples);

static uptr<Trainer> getTrainer(void);

void StackedAutoencoders::TrainAndEvaluate(void) {
  cout << "loading training data" << endl;
  vector<TrainingSample> trainingSamples =
      DataLoader::loadSamples("data/train_images.idx3", "data/train_labels.idx1", false);
  random_shuffle(trainingSamples.begin(), trainingSamples.end());
  cout << "training data size: " << trainingSamples.size() << endl;

  cout << "loading test data" << endl;
  vector<TrainingSample> testSamples =
      DataLoader::loadSamples("data/test_images.idx3", "data/test_labels.idx1", false);
  random_shuffle(testSamples.begin(), testSamples.end());
  cout << "test data size: " << testSamples.size() << endl;

  unsigned inputSize = trainingSamples.front().input.rows();
  unsigned outputSize = trainingSamples.front().expectedOutput.rows();

  Network featureExtractor = createFeatureExtractor(inputSize);
  Network classifier =
      createClassifier(featureExtractor.LayerSize(featureExtractor.NumLayers() - 1), outputSize);

  trainFeatureExtractor(featureExtractor, trainingSamples);

  vector<TrainingSample> processedTraining = featureExtract(featureExtractor, trainingSamples);
  vector<TrainingSample> processedTest = featureExtract(featureExtractor, testSamples);

  trainClassifier(classifier, processedTraining, processedTest);
}

Network createFeatureExtractor(unsigned inputSize) {
  auto hiddenActivation = make_shared<Logistic>();

  NetworkSpec spec;
  spec.numInputs = inputSize;
  spec.numOutputs = inputSize / 2;
  spec.outputFunc = make_shared<Logistic>();
  spec.hiddenLayers = {make_pair(inputSize, hiddenActivation)};

  return Network(spec);
}

Network createClassifier(unsigned inputSize, unsigned outputSize) {
  auto hiddenActivation = make_shared<ReLU>(0.01f);

  NetworkSpec spec;
  spec.numInputs = inputSize;
  spec.numOutputs = outputSize;
  spec.outputFunc = make_shared<Logistic>();
  spec.hiddenLayers = {make_pair(inputSize, hiddenActivation)};

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

void trainFeatureExtractor(Network &network, vector<TrainingSample> &trainingSamples) {
  cout << "training feature extractor" << endl;

  for (unsigned layer = 0; layer < network.NumLayers(); layer++) {
    cout << "conditioning layer: " << layer << endl;
    vector<TrainingSample> conditionSubsample =
        layer == 0 ? autoencodedSamplesFromTrainingData(trainingSamples)
                   : autoencodedSamplesFromNetworkLayer(network, layer, trainingSamples);

    // Autoencoder autoEncoder(0.25f);
    // Matrix hl = autoEncoder.ComputeHiddenLayer(network.LayerSize(layer), make_unique<Logistic>(),
    //                                            conditionSubsample, inputType);
    // network.SetLayerWeights(layer, hl);
  }

  cout << "finished training feature extractor" << endl;
}

vector<TrainingSample> featureExtract(Network &featureExtractor, vector<TrainingSample> &samples) {
  vector<TrainingSample> result;
  for (const auto &sample : samples) {
    result.emplace_back(featureExtractor.Process(sample.input), sample.expectedOutput);
  }
  return result;
}

void trainClassifier(Network &network, vector<TrainingSample> &trainingSamples,
                     vector<TrainingSample> &testSamples) {

  auto trainer = getTrainer();
  trainer->AddProgressCallback(
      [&trainingSamples, &testSamples](Network &network, float trainError, unsigned iter) {
        if (iter % 100 == 0) {
          float testWrong = ExperimentUtils::eval(network, testSamples);
          cout << iter << "\t" << testWrong << " " << trainError << endl;
          // cout << iter << "\t" << trainError << "\t" << testWrong << endl;

          // float trainWrong = testNetwork(network, trainingSamples);
          // cout << iter << "\t" << trainError << "\t" << testWrong << "\t" << trainWrong <<
          // endl;
        }
      });

  cout << "starting training..." << endl;

  // AllowSelectLayers restrictLayers({1});
  trainer->Train(network, trainingSamples, 20000, nullptr);
  cout << "finished" << endl;
}

uptr<Trainer> getTrainer(void) {
  return uptr<Trainer>(new AdamTrainer());
  // DynamicTrainerBuilder builder;
  //
  // builder.StartLearnRate(0.001f)
  //     .FinishLearnRate(0.0001f)
  //     .MaxLearnRate(0.1f)
  //     .Momentum(0.5f)
  //     .StartSamplesPerIter(5000)
  //     .FinishSamplesPerIter(5000)
  //     .UseMomentum(true)
  //     .UseSpeedup(true)
  //     .UseWeightRates(true);
  //
  // return builder.Build();
}
