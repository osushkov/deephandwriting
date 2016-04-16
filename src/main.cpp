
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

#include "ConditionedDeepNN.hpp"
#include "common/Common.hpp"

using Eigen::MatrixXd;

// void testAutoencoder(const vector<TrainingSample> &digitSamples) {
//   vector<TrainingSample> samples;
//   for (const auto &ds : digitSamples) {
//     // if (samples.size() > 1000) {
//     //   break;
//     // }
//     samples.push_back(TrainingSample(ds.input, ds.input));
//   }
//
//   Autoencoder encoder(0.25f);
//   encoder.ComputeHiddenLayer(700, make_unique<Logistic>(), samples,
//                              EncodedDataType::BOUNDED_NORMALISED);
// }

int main(int argc, char **argv) {
  Eigen::initParallel();
  srand(1234);

  // testAutoencoder(trainingSamples);
  ConditionedDeepNN::TrainAndEvaluate();

  return 0;
}
