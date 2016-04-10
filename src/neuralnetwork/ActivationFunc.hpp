
#pragma once

#include <cmath>

struct ActivationFunc {

  virtual ~ActivationFunc() = default;

  virtual float ActivationValue(float in) const = 0;
  virtual float DerivativeValue(float in, float out) const = 0;
};

struct Logistic : public ActivationFunc {
  float ActivationValue(float in) const override { return 1.0f / (1.0f + expf(-in)); }
  float DerivativeValue(float in, float out) const override { return out * (1.0f - out); }
};

struct Softplus : public ActivationFunc {
  float ActivationValue(float in) const override { return logf(1.0f + expf(in)); }
  float DerivativeValue(float in, float out) const override { return 1.0f / (1.0f + expf(-in)); }
};

struct ReLU : public ActivationFunc {
  ReLU() : leak(0.0f) {}
  ReLU(float leak) : leak(leak) {}

  float ActivationValue(float in) const override { return in > 0.0f ? in : (leak * in); }
  float DerivativeValue(float in, float out) const override { return in > 0.0f ? 1.0f : leak; }

private:
  const float leak;
};
