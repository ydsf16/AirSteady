#include "yolo/ort_backend.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

#if __has_include(<dml_provider_factory.h>)
#include <dml_provider_factory.h>
#define AIRSTEADY_HAS_DML 1
#elif __has_include(<onnxruntime/core/providers/dml/dml_provider_factory.h>)
#include <onnxruntime/core/providers/dml/dml_provider_factory.h>
#define AIRSTEADY_HAS_DML 1
#else
#define AIRSTEADY_HAS_DML 0
#endif

namespace airsteady {
namespace yolo {

static inline void SetErr(std::string* err, const std::string& msg) {
  if (err) *err = msg;
}

OrtBackend::OrtBackend()
    : env_(ORT_LOGGING_LEVEL_WARNING, "airsteady_yolo") {}

OrtBackend::~OrtBackend() = default;

std::vector<std::string> OrtBackend::GetAvailableProviders(std::string* err) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  char** providers = nullptr;
  int num = 0;
  OrtStatus* st = api->GetAvailableProviders(&providers, &num);
  if (st != nullptr) {
    const char* msg = api->GetErrorMessage(st);
    api->ReleaseStatus(st);
    SetErr(err, std::string("GetAvailableProviders failed: ") + (msg ? msg : ""));
    return {};
  }
  std::vector<std::string> out;
  out.reserve(static_cast<size_t>(num));
  for (int i = 0; i < num; ++i) out.emplace_back(providers[i] ? providers[i] : "");
  api->ReleaseAvailableProviders(providers, num);
  return out;
}

bool OrtBackend::ProviderAvailable(const char* provider_name, std::string* err) {
  auto ps = GetAvailableProviders(err);
  for (const auto& p : ps) {
    if (p == provider_name) return true;
  }
  return false;
}

bool OrtBackend::Init(const YoloConfig& cfg, Provider provider, std::string* err) {
  if (cfg.onnx_path.empty()) {
    SetErr(err, "onnx_path is empty.");
    return false;
  }

  so_ = Ort::SessionOptions{};
  if (cfg.intra_op_threads > 0) so_.SetIntraOpNumThreads(cfg.intra_op_threads);
  if (cfg.inter_op_threads > 0) so_.SetInterOpNumThreads(cfg.inter_op_threads);
  so_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  Provider target = provider;
  if (provider == Provider::kAuto) {
    // Windows: prefer DirectML if present, else CPU.
#if AIRSTEADY_HAS_DML
    std::string e;
    if (ProviderAvailable("DmlExecutionProvider", &e)) {
      target = Provider::kDirectML;
    } else {
      target = Provider::kCpu;
    }
#else
    target = Provider::kCpu;
#endif
  }

  bool ok = false;
  if (target == Provider::kDirectML) ok = InitDirectML(cfg, err);
  else if (target == Provider::kCpu) ok = InitCpu(cfg, err);
  else {
    SetErr(err, std::string("Provider not supported in this build: ") + ProviderToString(target));
    return false;
  }

  if (!ok) return false;

  // Create session
  try {
#ifdef _WIN32
    std::wstring wpath(cfg.onnx_path.begin(), cfg.onnx_path.end());
    session_ = std::make_unique<Ort::Session>(env_, wpath.c_str(), so_);
#else
    session_ = std::make_unique<Ort::Session>(env_, cfg.onnx_path.c_str(), so_);
#endif
  } catch (const std::exception& e) {
    SetErr(err, std::string("Create ORT session failed: ") + e.what());
    return false;
  }

  // IO names
  try {
    Ort::AllocatorWithDefaultOptions allocator;
    auto in_name = session_->GetInputNameAllocated(0, allocator);
    input_name_ = in_name.get() ? in_name.get() : "";

    const size_t out_count = session_->GetOutputCount();
    output_names_.clear();
    output_names_.reserve(out_count);
    for (size_t i = 0; i < out_count; ++i) {
      auto out_name = session_->GetOutputNameAllocated(i, allocator);
      output_names_.push_back(out_name.get() ? out_name.get() : "");
    }
  } catch (const std::exception& e) {
    SetErr(err, std::string("Get IO names failed: ") + e.what());
    return false;
  }

  return true;
}

bool OrtBackend::InitCpu(const YoloConfig& /*cfg*/, std::string* /*err*/) {
  provider_kind_ = Provider::kCpu;
  provider_name_ = "CPUExecutionProvider";
  return true;
}

bool OrtBackend::InitDirectML(const YoloConfig& cfg, std::string* err) {
#if !AIRSTEADY_HAS_DML
  SetErr(err, "DirectML headers not found. Use Microsoft.ML.OnnxRuntime.DirectML package (or equivalent).");
  return false;
#else
  std::string e;
  if (!ProviderAvailable("DmlExecutionProvider", &e)) {
    SetErr(err, e.empty() ? "DmlExecutionProvider not available." : e);
    return false;
  }

  // DirectML EP constraints: disable mem pattern and use sequential execution.
  so_.DisableMemPattern();
  so_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

  try {
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(so_, cfg.dml_device_id));
  } catch (const std::exception& ex) {
    SetErr(err, std::string("OrtSessionOptionsAppendExecutionProvider_DML failed: ") + ex.what());
    return false;
  }

  provider_kind_ = Provider::kDirectML;
  provider_name_ = "DmlExecutionProvider";
  return true;
#endif
}

bool OrtBackend::Run(const cv::Mat& blob_chw, std::vector<OrtOutput>* outs, std::string* err) {
  if (!session_) {
    SetErr(err, "ORT session is null. Call Init() first.");
    return false;
  }
  if (!outs) {
    SetErr(err, "outs is null.");
    return false;
  }
  outs->clear();

  if (blob_chw.empty() || blob_chw.type() != CV_32F || blob_chw.rows != 1) {
    SetErr(err, "blob_chw must be cv::Mat(1, 3*H*W, CV_32F).");
    return false;
  }

  const int64_t H = 1;  // infer from config? not stored; caller ensures correct size.
  (void)H;

  // The blob is [1, 3*in_h*in_w]. We need shape [1,3,in_h,in_w].
  // We cannot infer in_h/in_w uniquely from length without config; but session input shape is known:
  int64_t in_h = 0, in_w = 0;
  try {
    auto ti = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    auto s = ti.GetShape();
    // Expected [1,3,H,W] (could have -1 for dynamic)
    if (s.size() == 4) {
      in_h = (s[2] > 0) ? s[2] : 0;
      in_w = (s[3] > 0) ? s[3] : 0;
    }
  } catch (...) {
    // ignore
  }

  // Fallback: caller uses fixed cfg input_h/input_w, but we don't have it here.
  // If model is dynamic and we can't infer, we try to deduce from blob length:
  // len = 3 * h * w.
  const int64_t len = static_cast<int64_t>(blob_chw.cols);
  if (in_h <= 0 || in_w <= 0) {
    // Deduce a square-ish shape (works for common 640/1280).
    // Prefer to set static shapes in export to avoid ambiguity.
    const int64_t plane = len / 3;
    const int64_t root = static_cast<int64_t>(std::llround(std::sqrt(static_cast<double>(plane))));
    if (root * root == plane) {
      in_h = root;
      in_w = root;
    } else {
      SetErr(err, "Cannot infer input H/W from model shape. Export model with fixed input shape.");
      return false;
    }
  }

  const int64_t shape[4] = {1, 3, in_h, in_w};

  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

  const size_t numel = static_cast<size_t>(len);
  float* data = const_cast<float*>(blob_chw.ptr<float>(0));

  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      mem, data, numel, shape, 4);

  const char* in_names[] = {input_name_.c_str()};

  std::vector<const char*> out_names;
  out_names.reserve(output_names_.size());
  for (const auto& s : output_names_) out_names.push_back(s.c_str());

  try {
    auto out_tensors = session_->Run(Ort::RunOptions{nullptr},
                                     in_names, &input_tensor, 1,
                                     out_names.data(), out_names.size());

    outs->reserve(out_tensors.size());
    for (auto& t : out_tensors) {
      OrtOutput o;
      auto info = t.GetTensorTypeAndShapeInfo();
      o.shape = info.GetShape();

      const size_t out_numel = static_cast<size_t>(info.GetElementCount());
      o.data.resize(out_numel);

      const float* p = t.GetTensorData<float>();
      std::memcpy(o.data.data(), p, out_numel * sizeof(float));
      outs->push_back(std::move(o));
    }
  } catch (const std::exception& e) {
    SetErr(err, std::string("ORT Run failed: ") + e.what());
    return false;
  }

  return true;
}

}  // namespace yolo
}  // namespace airsteady
