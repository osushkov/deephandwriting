
#pragma once

#include "DynamicTrainer.hpp"
#include "common/Common.hpp"

class DynamicTrainerBuilder {
  float startLearnRate;
  float finishLearnRate;
  float maxLearnRate;

  float momentum;

  unsigned startNumSamples;
  unsigned finishNumSamples;

  bool useMomentum;
  bool useSpeedup;
  bool useWeightRates;

public:
  DynamicTrainerBuilder();
  uptr<DynamicTrainer> Build(void) const;

  DynamicTrainerBuilder& StartLearnRate(float);
  DynamicTrainerBuilder& FinishLearnRate(float);

  DynamicTrainerBuilder& MaxLearnRate(float);
  DynamicTrainerBuilder& Momentum(float);

  DynamicTrainerBuilder& StartSamplesPerIter(unsigned);
  DynamicTrainerBuilder& FinishSamplesPerIter(unsigned);

  DynamicTrainerBuilder &UseMomentum(bool);
  DynamicTrainerBuilder &UseSpeedup(bool);
  DynamicTrainerBuilder &UseWeightRates(bool);
};
