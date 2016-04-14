
#include "Autoencoder.hpp"
#include "../DynamicTrainer.hpp"
#include "../DynamicTrainerBuilder.hpp"
#include "../GradientRestriction.hpp"
#include "../image/CharImage.hpp"
#include "../image/IdxLabels.hpp"
#include "../image/ImageRenderer.hpp"
#include "../util/Util.hpp"
#include "Network.hpp"
#include <cassert>

class AutoencoderRestriction : public GradientRestriction {
public:
  void RestrictGradient(Tensor &gradient) {
    assert(gradient.NumLayers() == 2);
    assert(gradient(0).rows() == gradient(1).cols() - 1);
    assert(gradient(0).cols() == gradient(1).rows() + 1);

    Tensor orig = gradient;
    for (int r = 0; r < gradient(0).rows(); r++) {
      for (int c = 0; c < gradient(1).rows(); c++) {
        gradient(0)(r, c) = (orig(0)(r, c) + orig(1)(c, r)) / 2.0f;
        gradient(1)(c, r) = (orig(0)(r, c) + orig(1)(c, r)) / 2.0f;
      }
    }
  }
};

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
    // conditionInitialWeights(*network);

    auto trainer = getTrainer(samples.size());
    trainer->AddProgressCallback([](Network &network, float trainError, unsigned iter) {
      if (iter % 10 == 0) {
        cout << iter << "\t" << trainError << endl;
      }
    });

    // AutoencoderRestriction restriction;
    vector<TrainingSample> noisySamples = getNoisySamples(samples, dataType);
    trainer->Train(*(network.get()), noisySamples, 1000, nullptr);

    // debugNetworkVisually(network.get(), noisySamples);
    return network->LayerWeights(0);
  }

  void debugNetworkVisually(Network *network, const vector<TrainingSample> &noisySamples) {
    cout << "woo" << endl;
    for (const auto &sample : noisySamples) {
      Vector output = network->Process(sample.input);

      CharImage inImg(28, 28, sampleToVector(sample.input));
      ImageRenderer::RenderImage(inImg);
      // getchar();

      CharImage outImg(28, 28, sampleToVector(output));
      ImageRenderer::RenderImage(outImg);
      // getchar();
    }
    cout << "shit" << endl;
  }

  vector<float> sampleToVector(const Vector &s) {
    vector<float> result;
    for (int i = 0; i < s.rows(); i++) {
      result.push_back(fabs(s(i)));
    }
    return result;
  }

  void conditionInitialWeights(Network &network) {
    Matrix w0 = network.LayerWeights(0);
    Matrix w1 = network.LayerWeights(1);

    for (int r = 0; r < min(w0.rows(), w1.cols()); r++) {
      for (int c = 0; c < min(w0.cols(), w1.rows()); c++) {
        w0(r, c) = w1(c, r);
      }
    }

    network.SetLayerWeights(0, w0);
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

    builder.StartLearnRate(0.01f)
        .FinishLearnRate(0.0001f)
        .MaxLearnRate(0.1f)
        .Momentum(0.25f)
        .StartSamplesPerIter(1000)
        .FinishSamplesPerIter(1000)
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
