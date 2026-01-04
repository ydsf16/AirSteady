#pragma once

#include <opencv2/core.hpp>

#include "yolo/yolo_types.h"

namespace airsteady {
namespace yolo {

// Output blob: cv::Mat(1, 3*H*W, CV_32F), CHW, range [0,1], BGR order.
cv::Mat LetterBoxBgrToBlobCHW(const cv::Mat& bgr,
                              int dst_w,
                              int dst_h,
                              bool keep_aspect,
                              LetterBoxInfo* info);

}  // namespace yolo
}  // namespace airsteady
