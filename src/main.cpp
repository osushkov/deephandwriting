
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

#include "DataLoader.hpp"
#include "common/Common.hpp"
#include "learningpolicies/DynamicTrainer.hpp"
#include "learningpolicies/DynamicTrainerBuilder.hpp"
#include "learningpolicies/SimpleTrainer.hpp"
#include "neuralnetwork/Autoencoder.hpp"
#include "neuralnetwork/Network.hpp"
#include "util/Timer.hpp"
#include "util/Util.hpp"

using namespace std;
using Eigen::MatrixXd;

int digitFromNNOutput(const Vector &out) {
  assert(out.rows() == 10);

  int result = 0;
  float maxActivation = out(0);

  for (int i = 1; i < out.rows(); i++) {
    if (out(i) > maxActivation) {
      maxActivation = out(i);
      result = i;
    }
  }

  return result;
}

float eval(Network &network, const vector<TrainingSample> &testSamples) {
  unsigned numCorrect = 0;

  for (const auto &ts : testSamples) {
    auto result = network.Process(ts.input);
    bool isCorrect = digitFromNNOutput(result) == digitFromNNOutput(ts.expectedOutput);
    numCorrect += isCorrect ? 1 : 0;
  }

  return 1.0f - numCorrect / static_cast<float>(testSamples.size());
}

Network createNewNetwork(unsigned inputSize, unsigned outputSize) {
  auto hiddenActivation = make_shared<Logistic>();

  NetworkSpec spec;
  spec.numInputs = inputSize;
  spec.numOutputs = outputSize;
  spec.outputFunc = make_shared<Logistic>();
  // spec.hiddenLayers = {make_pair(inputSize / 4, hiddenActivation)};
  spec.hiddenLayers = {make_pair(inputSize, hiddenActivation),
                       make_pair(inputSize / 2, hiddenActivation),
                       make_pair(inputSize / 4, hiddenActivation)};

  return Network(spec);
}

Network loadNetwork(string path) {
  ifstream networkIn(path, ios::in | ios::binary);
  return Network(networkIn);
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

    EncodedDataType inputType = EncodedDataType::BOUNDED_NORMALISED;
    // layer == 0 ? EncodedDataType::BOUNDED_NORMALISED : EncodedDataType::UNBOUNDED;

    Autoencoder autoEncoder(0.25f);
    Matrix hl = autoEncoder.ComputeHiddenLayer(network.LayerSize(layer), make_unique<Logistic>(),
                                               conditionSubsample, inputType);
    network.SetLayerWeights(layer, hl);
  }

  cout << "finished conditioning" << endl;
}

uptr<Trainer> getTrainer(void) {
  DynamicTrainerBuilder builder;

  builder.StartLearnRate(0.01f)
      .FinishLearnRate(0.001f)
      .MaxLearnRate(0.1f)
      .Momentum(0.25f)
      .StartSamplesPerIter(5000)
      .FinishSamplesPerIter(5000)
      .UseMomentum(true)
      .UseSpeedup(true)
      .UseWeightRates(true);

  return builder.Build();
}

void learn(Network &network, vector<TrainingSample> &trainingSamples,
           vector<TrainingSample> &testSamples) {
  conditionNetwork(network, trainingSamples);

  auto trainer = getTrainer();
  trainer->AddProgressCallback(
      [&trainingSamples, &testSamples](Network &network, float trainError, unsigned iter) {
        if (iter % 100 == 0) {
          float testWrong = eval(network, testSamples);
          cout << iter << "\t" << testWrong << " " << trainError << endl;
          // cout << iter << "\t" << trainError << "\t" << testWrong << endl;

          // float trainWrong = testNetwork(network, trainingSamples);
          // cout << iter << "\t" << trainError << "\t" << testWrong << "\t" << trainWrong << endl;
        }
      });

  cout << "starting training..." << endl;

  // AllowSelectLayers restrictLayers({1});
  trainer->Train(network, trainingSamples, 20000, nullptr);
  cout << "finished" << endl;
}

void testAutoencoder(const vector<TrainingSample> &digitSamples) {
  vector<TrainingSample> samples;
  for (const auto &ds : digitSamples) {
    // if (samples.size() > 1000) {
    //   break;
    // }
    samples.push_back(TrainingSample(ds.input, ds.input));
  }

  Autoencoder encoder(0.25f);
  encoder.ComputeHiddenLayer(700, make_unique<Logistic>(), samples,
                             EncodedDataType::BOUNDED_NORMALISED);
}

int main(int argc, char **argv) {
  Eigen::initParallel();
  srand(1234);

  // TODO: training+test image data paths can be command line args.

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

  // testAutoencoder(trainingSamples);

  unsigned inputSize = trainingSamples.front().input.rows();
  unsigned outputSize = trainingSamples.front().expectedOutput.rows();

  // TODO: should probably use a command line args parsing library here.
  if (argc == 1 || (argc >= 2 && string(argv[1]) == "train")) {
    Network network = argc == 3 ? loadNetwork(argv[2]) : createNewNetwork(inputSize, outputSize);
    learn(network, trainingSamples, testSamples);

    ofstream networkOut("network.dat", ios::out | ios::binary);
    network.Serialize(networkOut);
  } else if (argc == 3 && string(argv[1]) == "eval") {
    Network network = loadNetwork(argv[2]);
    float errorRate = eval(network, testSamples);
    cout << endl << "error rate: " << (100.0f * (1.0f - errorRate)) << endl;
  } else {
    cout << "invalid arguments, expected: " << endl;
    cout << string(argv[0]) << " train [existing_network_file]" << endl;
    cout << string(argv[0]) << " test network_file" << endl;
  }

  return 0;
}
