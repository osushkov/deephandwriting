
#include "ImageWriter.hpp"
#include <opencv2/opencv.hpp>

using Vec1b = cv::Vec<unsigned char, 1>;

void ImageWriter::WriteImage(const CharImage &img, string outPath) {
  cv::Mat outImg(img.width, img.height, CV_8UC1);

  for (int y = 0; y < outImg.rows; y++) {
    for (int x = 0; x < outImg.cols; x++) {
      Vec1b &v = outImg.at<Vec1b>(y, x);
      v[0] = static_cast<unsigned char>(img.pixels[x + y * img.width] * 255);
    }
  }

  cv::imwrite(outPath, outImg);
}
