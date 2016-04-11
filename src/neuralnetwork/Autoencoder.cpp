
#include "Autoencoder.hpp"
#include "../DynamicTrainer.hpp"
#include "../DynamicTrainerBuilder.hpp"
#include "../util/Util.hpp"
#include "Network.hpp"
#include <cassert>

struct Autoencoder::AutoencoderImpl {

  const float pLoss;

  AutoencoderImpl(float pLoss) : pLoss(pLoss) { assert(pLoss <= 1.0f && pLoss >= 0.0f); }

  Matrix ComputeHiddenLayer(unsigned layerSize, uptr<ActivationFunc> hiddenFunc,
                            const vector<TrainingSample> &samples, EncodedDataType dataType) {

    assert(layerSize > 0);
    assert(!samples.empty());

    NetworkSpec spec;
    spec.numInputs = samples[0].input.rows();
    spec.numOutputs = spec.numInputs;
    spec.outputFunc = activationFuncForData(dataType);
    spec.hiddenLayers = {make_pair(layerSize, move(hiddenFunc))};

    auto network = make_unique<Network>(spec);
    auto trainer = getTrainer(samples.size());
    trainer->AddProgressCallback([](Network &network, float trainError, unsigned iter) {
      if (iter % 10 == 0) {
        cout << iter << "\t" << trainError << endl;
      }
    });

    vector<TrainingSample> noisySamples = getNoisySamples(samples, dataType);
    trainer->Train(*(network.get()), noisySamples, 1000);
    return network->LayerWeights(0);
  }

  vector<TrainingSample> getNoisySamples(const vector<TrainingSample> &samples,
                                         EncodedDataType dataType) {
    vector<TrainingSample> noisySamples;
    noisySamples.reserve(samples.size());

    for (const auto &s : samples) {
      noisySamples.push_back(getNoisySample(s, dataType));
    }

    return noisySamples;
  }

  TrainingSample getNoisySample(const TrainingSample &sample, EncodedDataType dataType) {
    Vector input = sample.input;
    for (int r = 0; r < input.rows(); r++) {
      if (Util::RandInterval(0.0, 1.0) < pLoss) {
        input(r) = 0.0f;
      }
    }
    return TrainingSample(input, sample.expectedOutput);
  }

  uptr<Trainer> getTrainer(unsigned numSamples) {
    DynamicTrainerBuilder builder;

    builder.StartLearnRate(0.1f)
        .FinishLearnRate(0.01f)
        .MaxLearnRate(0.1f)
        .Momentum(0.5f)
        .StartSamplesPerIter(numSamples / 20)
        .FinishSamplesPerIter(numSamples / 10)
        .UseMomentum(true)
        .UseSpeedup(true)
        .UseWeightRates(true);

    return builder.Build();
  }

  uptr<ActivationFunc> activationFuncForData(EncodedDataType dataType) {
    if (dataType == EncodedDataType::BOUNDED_NORMALISED) {
      return make_unique<Logistic>();
    } else {
      return make_unique<Linear>();
    }
  }
};

Autoencoder::Autoencoder(float pLoss) : impl(new AutoencoderImpl(pLoss)) {}
Autoencoder::~Autoencoder() = default;

Matrix Autoencoder::ComputeHiddenLayer(unsigned layerSize, uptr<ActivationFunc> hiddenFunc,
                                       const vector<TrainingSample> &samples,
                                       EncodedDataType dataType) {
  return impl->ComputeHiddenLayer(layerSize, move(hiddenFunc), samples, dataType);
}
