
#pragma once

#include "../common/Math.hpp"
#include <vector>

namespace Util {
double RandInterval(double s, double e);
double GaussianSample(double mean, double sd);
Vector SoftmaxActivations(const Vector &in);
}
