#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "yolo/yolo_types.h"

namespace airsteady {
namespace yolo {

class YoloSegDetector {
 public:
  explicit YoloSegDetector(const YoloConfig& cfg);
  ~YoloSegDetector();

  YoloSegDetector(const YoloSegDetector&) = delete;
  YoloSegDetector& operator=(const YoloSegDetector&) = delete;

  void SetEventCallback(EventCallback cb);

  RuntimeStatus GetStatus() const;

  bool Init(std::string* err);

  bool Infer(const cv::Mat& bgr, YoloResult* out, std::string* err);

  bool InferBatch(const std::vector<cv::Mat>& bgr_list,
                  std::vector<YoloResult>* out_list,
                  std::string* err);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace yolo
}  // namespace airsteady
