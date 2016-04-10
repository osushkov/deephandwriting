#pragma once

#include "../common/Common.hpp"
#include "CharImage.hpp"

class ImageWriter {
public:
  void WriteImage(const CharImage &img, string outPath);
};
