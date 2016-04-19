
#include "Util.hpp"
#include "../common/Math.hpp"
#include <cmath>
#include <cstdlib>

double Util::RandInterval(double s, double e) { return s + (e - s) * (rand() / (double)RAND_MAX); }

double Util::GaussianSample(double mean, double sd) {
  // Taken from GSL Library Gaussian random distribution.
  double x, y, r2;

  do {
    // choose x,y in uniform square (-1,-1) to (+1,+1)
    x = -1 + 2 * RandInterval(0.0, 1.0);
    y = -1 + 2 * RandInterval(0.0, 1.0);

    // see if it is in the unit circle
    r2 = x * x + y * y;
  } while (r2 > 1.0 || r2 == 0);

  // Box-Muller transform
  return mean + sd * y * sqrt(-2.0 * log(r2) / r2);
}

Vector Util::SoftmaxActivations(const Vector &in) {
  assert(in.rows() > 0);
  Vector result(in.rows());

  float maxVal = in(0);
  for (int r = 0; r < in.rows(); r++) {
    maxVal = fmax(maxVal, in(r));
  }

  float sum = 0.0f;
  for (int i = 0; i < in.rows(); i++) {
    result(i) = expf(in(i)-maxVal);
    sum += result(i);
  }

  for (int i = 0; i < result.rows(); i++) {
    result(i) /= sum;
  }

  return result;
}
