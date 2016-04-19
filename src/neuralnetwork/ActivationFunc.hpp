
#pragma once

#include "../common/Common.hpp"
#include <cmath>

struct ActivationFunc {
  virtual ~ActivationFunc() = default;

  virtual float ActivationValue(float in) const = 0;
  virtual float DerivativeValue(float in, float out) const = 0;
  virtual uptr<ActivationFunc> Clone(void) const = 0;
};

struct Logistic : public ActivationFunc {
  float ActivationValue(float in) const override { return 1.0f / (1.0f + expf(-in)); }
  float DerivativeValue(float in, float out) const override { return out * (1.0f - out); }
  uptr<ActivationFunc> Clone(void) const override { return make_unique<Logistic>(); }
};

struct Tanh : public ActivationFunc {
  float ActivationValue(float in) const override { return tanh(in); }
  float DerivativeValue(float in, float out) const override { return 1.0f - out * out; }
  uptr<ActivationFunc> Clone(void) const override { return make_unique<Tanh>(); }
};

struct Softplus : public ActivationFunc {
  float ActivationValue(float in) const override { return logf(1.0f + expf(in)); }
  float DerivativeValue(float in, float out) const override { return 1.0f / (1.0f + expf(-in)); }
  uptr<ActivationFunc> Clone(void) const override { return make_unique<Softplus>(); }
};

struct ReLU : public ActivationFunc {
  ReLU() : leak(0.0f) {}
  ReLU(float leak) : leak(leak) {}

  float ActivationValue(float in) const override { return in > 0.0f ? in : (leak * in); }
  float DerivativeValue(float in, float out) const override { return in > 0.0f ? 1.0f : leak; }
  uptr<ActivationFunc> Clone(void) const override { return make_unique<ReLU>(leak); }

private:
  const float leak;
};

struct Linear : public ActivationFunc {
  float ActivationValue(float in) const override { return in; }
  float DerivativeValue(float in, float out) const override { return 1.0f; }
  uptr<ActivationFunc> Clone(void) const override { return make_unique<Linear>(); }
};
