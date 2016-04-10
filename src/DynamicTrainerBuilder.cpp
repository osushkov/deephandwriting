
#include "DynamicTrainerBuilder.hpp"
#include "common/Common.hpp"

#include <cassert>

DynamicTrainerBuilder::DynamicTrainerBuilder()
    : startLearnRate(0.5f), finishLearnRate(0.001f), maxLearnRate(0.5f), momentum(0.25f) {}

uptr<DynamicTrainer> DynamicTrainerBuilder::Build(void) const {
  assert(startLearnRate > 0.0f);
  assert(finishLearnRate > 0.0f);
  assert(maxLearnRate > 0.0f);
  assert(startLearnRate >= finishLearnRate);
  assert(maxLearnRate >= startLearnRate && maxLearnRate >= finishLearnRate);
  assert(momentum >= 0.0f);
  assert(startNumSamples > 0 && finishLearnRate > 0);

  return make_unique<DynamicTrainer>(startLearnRate, finishLearnRate, maxLearnRate, momentum,
                                     startNumSamples, finishNumSamples, useMomentum, useSpeedup,
                                     useWeightRates);
}

DynamicTrainerBuilder &DynamicTrainerBuilder::StartLearnRate(float lr) {
  this->startLearnRate = lr;
  return *this;
}

DynamicTrainerBuilder &DynamicTrainerBuilder::FinishLearnRate(float lr) {
  this->finishLearnRate = lr;
  return *this;
}

DynamicTrainerBuilder &DynamicTrainerBuilder::MaxLearnRate(float lr) {
  this->maxLearnRate = lr;
  return *this;
}

DynamicTrainerBuilder &DynamicTrainerBuilder::Momentum(float m) {
  this->momentum = m;
  return *this;
}

DynamicTrainerBuilder &DynamicTrainerBuilder::StartSamplesPerIter(unsigned samples) {
  this->startNumSamples = samples;
  return *this;
}

DynamicTrainerBuilder &DynamicTrainerBuilder::FinishSamplesPerIter(unsigned samples) {
  this->finishNumSamples = samples;
  return *this;
}

DynamicTrainerBuilder &DynamicTrainerBuilder::UseMomentum(bool u) {
  this->useMomentum = u;
  return *this;
}

DynamicTrainerBuilder &DynamicTrainerBuilder::UseSpeedup(bool u) {
  this->useSpeedup = u;
  return *this;
}

DynamicTrainerBuilder &DynamicTrainerBuilder::UseWeightRates(bool u) {
  this->useWeightRates = u;
  return *this;
}
