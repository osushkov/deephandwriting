
#include "ImageRenderer.hpp"
#include "../common/Common.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

using Vec1b = cv::Vec<unsigned char, 1>;

// TODO: this is copy pasted from ImageGEnerator
static cv::Mat convertToMat(const CharImage &img) {
  cv::Mat outImg(img.width, img.height, CV_8UC1);

  for (int y = 0; y < outImg.rows; y++) {
    for (int x = 0; x < outImg.cols; x++) {
      Vec1b &v = outImg.at<Vec1b>(y, x);
      v[0] = static_cast<unsigned char>(img.pixels[x + y * img.width] * 255);
    }
  }

  return outImg;
}

void ImageRenderer::RenderImage(CharImage &img) {
  cout << "render image" << endl;
  cv::namedWindow("Image Window", cv::WINDOW_AUTOSIZE);

  auto mat = convertToMat(img);
  cv::imshow("Image Window", mat);
  cv::waitKey(0);
}
