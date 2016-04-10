
#include "Autoencoder.hpp"
#include "../DynamicTrainer.hpp"
#include "../DynamicTrainerBuilder.hpp"
#include "Network.hpp"
#include <cassert>

struct Autoencoder::AutoencoderImpl {
  uptr<Network> network;

  AutoencoderImpl(unsigned inputSize, unsigned hiddenSize) {
    assert(inputSize > 0 && hiddenSize > 0);

    NetworkSpec spec;
    spec.numInputs = inputSize;
    spec.numOutputs = inputSize;
    // spec.outputFunc

    network = make_unique<Network>(spec);
  }

  Matrix ComputeHiddenLayer(vector<TrainingSample> &samples) {
    auto trainer = getTrainer(samples.size());
    trainer->Train(*(network.get()), samples, 1000);
    return network->LayerWeights(0);
  }

  uptr<Trainer> getTrainer(unsigned numSamples) {
    DynamicTrainerBuilder builder;

    builder.StartLearnRate(0.5f)
        .FinishLearnRate(0.01f)
        .MaxLearnRate(0.5f)
        .Momentum(0.5f)
        .StartSamplesPerIter(numSamples / 100)
        .FinishSamplesPerIter(numSamples / 10)
        .UseMomentum(true)
        .UseSpeedup(true)
        .UseWeightRates(true);

    return builder.Build();
  }
};

Autoencoder::Autoencoder(unsigned inputSize, unsigned hiddenSize)
    : impl(new AutoencoderImpl(inputSize, hiddenSize)) {}

Autoencoder::~Autoencoder() = default;

Matrix Autoencoder::ComputeHiddenLayer(vector<TrainingSample> &samples) {
  return impl->ComputeHiddenLayer(samples);
}
