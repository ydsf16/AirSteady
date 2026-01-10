#pragma once

#include <string>

#include <opencv2/core.hpp>

#include "yolo/ort_backend.h"
#include "yolo/yolo_types.h"

namespace airsteady {
namespace yolo {

// Supports common Ultralytics seg ONNX outputs:
// - dets: [1,C,N] or [1,N,C], where C = 4 + nc + mask_dim
// - proto: [1,mask_dim,mh,mw] (common: [1,32,160,160])
bool PostprocessYoloSeg(const YoloConfig& cfg,
                        const cv::Size& src_size,
                        const LetterBoxInfo& lb,
                        const OrtOutput& det_out,
                        const OrtOutput& proto_out,
                        YoloResult* out,
                        std::string* err);

}  // namespace yolo
}  // namespace airsteady
