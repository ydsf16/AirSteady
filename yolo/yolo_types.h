#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace airsteady {
namespace yolo {

enum class Provider {
  kAuto,
  kDirectML,  // Windows
  kCoreML,    // macOS (预留)
  kCpu,
};

inline const char* ProviderToString(Provider p) {
  switch (p) {
    case Provider::kAuto:     return "Auto";
    case Provider::kDirectML: return "DirectML";
    case Provider::kCoreML:   return "CoreML";
    case Provider::kCpu:      return "CPU";
    default:                  return "Unknown";
  }
}

struct YoloConfig {
  std::string onnx_path;

  int input_w = 640;
  int input_h = 640;
  bool keep_aspect_letterbox = true;

  Provider preferred_provider = Provider::kAuto;
  bool enable_auto_fallback = true;

  // Thresholds / limits
  float conf_thresh = 0.25f;
  float iou_thresh = 0.45f;
  int top_k = 100;
  bool class_agnostic_nms = false;

  // Mask (seg)
  float mask_thresh = 0.5f;   // binarize after sigmoid
  bool return_masks = true;   // you can disable for speed

  // ORT threading (CPU EP mainly)
  int intra_op_threads = 0;  // 0 => ORT default
  int inter_op_threads = 0;

  // DirectML options (Windows)
  int dml_device_id = 0;

  // Logging / profiling
  bool verbose_log = true;
  bool enable_profiling = true;
  int profile_every = 30;  // print every N inferences
};

struct InferenceStats {
  double preprocess_ms = 0.0;
  double infer_ms = 0.0;
  double postprocess_ms = 0.0;
  double total_ms = 0.0;
};

struct RuntimeStatus {
  std::string active_provider;  // "DmlExecutionProvider" / "CPUExecutionProvider"
  bool is_fallback = false;
  std::string last_error;
};

enum class EventSeverity { kInfo, kWarning, kError };
enum class EventCode {
  kInitStarted,
  kUsingProvider,
  kProviderInitFailed,
  kFallbackToCpu,
  kInitSucceeded,
  kInitFailed,
  kInferFailed,
  kPerf,
};

struct DetectorEvent {
  EventSeverity severity;
  EventCode code;
  std::string message;

  std::string details;
  std::string from_provider;
  std::string to_provider;
};

using EventCallback = std::function<void(const DetectorEvent&)>;

struct LetterBoxInfo {
  float scale = 1.0f;
  int pad_x = 0;
  int pad_y = 0;
  int in_w = 0;
  int in_h = 0;
  int resized_w = 0;
  int resized_h = 0;
};

struct Det {
  int class_id = -1;
  float score = 0.0f;
  // bbox in original image coordinates (x1,y1,x2,y2)
  cv::Rect box;

  // Optional mask in original image size: CV_8UC1, 0/255
  cv::Mat mask;
};

struct YoloResult {
  std::vector<Det> dets;

  // Optional debug info
  bool has_letterbox = false;
  LetterBoxInfo letterbox;

  bool has_stats = false;
  InferenceStats stats;
};

}  // namespace yolo
}  // namespace airsteady
