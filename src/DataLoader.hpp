#pragma once

#include <map>
#include <vector>

#include "neuralnetwork/TrainingSample.hpp"

namespace DataLoader {
vector<TrainingSample> loadSamples(string inImagePath, string inLabelPath, bool genDerived);
}
