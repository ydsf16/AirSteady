#include "yolo/letterbox.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace airsteady {
namespace yolo {

cv::Mat LetterBoxBgrToBlobCHW(const cv::Mat& bgr,
                              int dst_w,
                              int dst_h,
                              bool keep_aspect,
                              LetterBoxInfo* info) {
  CV_Assert(!bgr.empty());
  CV_Assert(bgr.type() == CV_8UC3);
  CV_Assert(dst_w > 0 && dst_h > 0);

  const int src_w = bgr.cols;
  const int src_h = bgr.rows;

  int new_w = dst_w;
  int new_h = dst_h;
  float scale = 1.0f;

  if (keep_aspect) {
    scale = std::min(static_cast<float>(dst_w) / static_cast<float>(src_w),
                     static_cast<float>(dst_h) / static_cast<float>(src_h));
    new_w = static_cast<int>(std::round(src_w * scale));
    new_h = static_cast<int>(std::round(src_h * scale));
  } else {
    scale = static_cast<float>(dst_w) / static_cast<float>(src_w);
    new_w = dst_w;
    new_h = dst_h;
  }

  const int pad_x = (dst_w - new_w) / 2;
  const int pad_y = (dst_h - new_h) / 2;

  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

  cv::Mat padded(dst_h, dst_w, CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(padded(cv::Rect(pad_x, pad_y, new_w, new_h)));

  if (info) {
    info->scale = scale;
    info->pad_x = pad_x;
    info->pad_y = pad_y;
    info->in_w = dst_w;
    info->in_h = dst_h;
    info->resized_w = new_w;
    info->resized_h = new_h;
  }

  cv::Mat float_img;
  padded.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

  std::vector<cv::Mat> chw(3);
  cv::split(float_img, chw);

  cv::Mat blob(1, 3 * dst_h * dst_w, CV_32F);
  float* p = blob.ptr<float>(0);
  const int plane = dst_h * dst_w;
  for (int c = 0; c < 3; ++c) {
    std::memcpy(p + c * plane, chw[c].ptr<float>(0),
                static_cast<size_t>(plane) * sizeof(float));
  }
  return blob;
}

}  // namespace yolo
}  // namespace airsteady
