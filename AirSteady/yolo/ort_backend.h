#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include <onnxruntime_cxx_api.h>

#include "yolo/yolo_types.h"

namespace airsteady {
namespace yolo {

struct OrtOutput {
  std::vector<int64_t> shape;
  std::vector<float> data;  // copied out (lifetime safe)
};

class OrtBackend {
 public:
  OrtBackend();
  ~OrtBackend();

  OrtBackend(const OrtBackend&) = delete;
  OrtBackend& operator=(const OrtBackend&) = delete;

  bool Init(const YoloConfig& cfg, Provider provider, std::string* err);

  // Input blob is CHW float: cv::Mat(1, 3*H*W, CV_32F)
  bool Run(const cv::Mat& blob_chw, std::vector<OrtOutput>* outs, std::string* err);

  const std::string& provider_name() const { return provider_name_; }
  Provider provider_kind() const { return provider_kind_; }

  static std::vector<std::string> GetAvailableProviders(std::string* err);
  static bool ProviderAvailable(const char* provider_name, std::string* err);

 private:
  bool InitCpu(const YoloConfig& cfg, std::string* err);
  bool InitDirectML(const YoloConfig& cfg, std::string* err);

 private:
  Provider provider_kind_ = Provider::kCpu;
  std::string provider_name_ = "CPUExecutionProvider";

  Ort::Env env_;
  Ort::SessionOptions so_;
  std::unique_ptr<Ort::Session> session_;

  std::string input_name_;
  std::vector<std::string> output_names_;
};

}  // namespace yolo
}  // namespace airsteady
