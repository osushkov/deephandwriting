
#include "Tensor.hpp"
#include <algorithm>
#include <cassert>
#include <iostream>

unsigned Tensor::NumLayers(void) const { return this->data.size(); }

void Tensor::AddLayer(const Matrix &m) { this->data.push_back(m); }

void Tensor::Deserialize(istream &stream) {
  data.clear();

  unsigned numLayers = 0;
  stream.read((char *)&numLayers, sizeof(unsigned));

  data.reserve(numLayers);
  for (unsigned i = 0; i < numLayers; i++) {
    int rows, cols;

    stream.read((char *)&rows, sizeof(int));
    stream.read((char *)&cols, sizeof(int));
    assert(rows > 0 && cols > 0);

    Matrix m(rows, cols);
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        float v;
        stream.read((char *)&v, sizeof(float));
        m(r, c) = v;
      }
    }

    AddLayer(m);
  }
}

void Tensor::Serialize(ostream &stream) const {
  unsigned numLayers = NumLayers();
  stream.write((char *)&numLayers, sizeof(unsigned));

  for (const auto &m : data) {
    int rows = m.rows();
    int cols = m.cols();

    stream.write((char *)&rows, sizeof(int));
    stream.write((char *)&cols, sizeof(int));

    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        float v = m(r, c);
        stream.write((char *)&v, sizeof(float));
      }
    }
  }
}

Matrix &Tensor::operator()(unsigned index) {
  assert(index < data.size());
  return data[index];
}

const Matrix &Tensor::operator()(unsigned index) const {
  assert(index < data.size());
  return data[index];
}

Tensor Tensor::operator*(const Tensor &t) const {
  Tensor result(*this);
  result *= t;
  return result;
}

Tensor Tensor::operator+(const Tensor &t) const {
  Tensor result(*this);
  result += t;
  return result;
}

Tensor Tensor::operator-(const Tensor &t) const {
  Tensor result(*this);
  result -= t;
  return result;
}

Tensor Tensor::operator*(float s) const {
  Tensor result(*this);
  result *= s;
  return result;
}

Tensor Tensor::operator/(float s) const {
  Tensor result(*this);
  result /= s;
  return result;
}

Tensor &Tensor::operator*=(const Tensor &t) {
  assert(this->NumLayers() == t.NumLayers());
  for (unsigned i = 0; i < NumLayers(); i++) {
    assert(data[i].rows() == t.data[i].rows());
    assert(data[i].cols() == t.data[i].cols());

    for (int y = 0; y < data[i].rows(); y++) {
      for (int x = 0; x < data[i].cols(); x++) {
        data[i](y, x) *= t.data[i](y, x);
      }
    }
  }
  return *this;
}

Tensor &Tensor::operator+=(const Tensor &t) {
  assert(this->NumLayers() == t.NumLayers());
  for (unsigned i = 0; i < NumLayers(); i++) {
    data[i] += t.data[i];
  }
  return *this;
}

Tensor &Tensor::operator-=(const Tensor &t) {
  assert(this->NumLayers() == t.NumLayers());
  for (unsigned i = 0; i < NumLayers(); i++) {
    data[i] -= t.data[i];
  }
  return *this;
}

Tensor &Tensor::operator*=(float s) {
  for_each(data, [=](Matrix &m) { m *= s; });
  return *this;
}

Tensor &Tensor::operator/=(float s) {
  float inv = 1.0f / s;
  for_each(data, [=](Matrix &m) { m *= inv; });
  return *this;
}
