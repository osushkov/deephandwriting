
#pragma once

#include "neuralnetwork/Network.hpp"
#include "neuralnetwork/TrainingSample.hpp"
#include <vector>

namespace ExperimentUtils {
float eval(Network &network, const vector<TrainingSample> &testSamples);
}
