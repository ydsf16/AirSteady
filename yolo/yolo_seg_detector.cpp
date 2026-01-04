#include "yolo/yolo_seg_detector.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include <glog/logging.h>

#include "yolo/letterbox.h"
#include "yolo/ort_backend.h"
#include "yolo/postprocess.h"
#include "yolo/time_utils.h"

namespace airsteady {
namespace yolo {

namespace {

// ---------- Simple percentile helper (copy-safe, O(n log n)) ----------
static double Percentile(std::vector<double> v, double p01) {
  if (v.empty()) return 0.0;
  if (p01 <= 0.0) return *std::min_element(v.begin(), v.end());
  if (p01 >= 1.0) return *std::max_element(v.begin(), v.end());
  std::sort(v.begin(), v.end());
  const double idx = (static_cast<double>(v.size()) - 1.0) * p01;
  const size_t i0 = static_cast<size_t>(std::floor(idx));
  const size_t i1 = std::min(i0 + 1, v.size() - 1);
  const double t = idx - static_cast<double>(i0);
  return v[i0] * (1.0 - t) + v[i1] * t;
}

struct PerfStats {
  // keep history for percentiles; if you worry about memory, switch to reservoir sampling later
  std::vector<double> pre_ms;
  std::vector<double> infer_ms;
  std::vector<double> post_ms;
  std::vector<double> total_ms;

  double min_total = std::numeric_limits<double>::infinity();
  double max_total = 0.0;

  void Add(const InferenceStats& s) {
    pre_ms.push_back(s.preprocess_ms);
    infer_ms.push_back(s.infer_ms);
    post_ms.push_back(s.postprocess_ms);
    total_ms.push_back(s.total_ms);
    min_total = std::min(min_total, s.total_ms);
    max_total = std::max(max_total, s.total_ms);
  }

  uint64_t Num() const { return static_cast<uint64_t>(total_ms.size()); }

  double Avg(const std::vector<double>& v) const {
    if (v.empty()) return 0.0;
    double sum = 0.0;
    for (double x : v) sum += x;
    return sum / static_cast<double>(v.size());
  }

  void LogSummary(const std::string& tag, const std::string& provider) const {
    const uint64_t n = Num();
    if (n == 0) {
      LOG(INFO) << "[YOLO][Perf] " << tag << " provider=" << provider << " (no runs)";
      return;
    }

    const double avg_pre = Avg(pre_ms);
    const double avg_inf = Avg(infer_ms);
    const double avg_post = Avg(post_ms);
    const double avg_tot = Avg(total_ms);

    const double p50 = Percentile(total_ms, 0.50);
    const double p90 = Percentile(total_ms, 0.90);
    const double p99 = Percentile(total_ms, 0.99);

    LOG(INFO) << "[YOLO][Perf] " << tag
              << " provider=" << provider
              << " runs=" << n
              << " avg(ms) pre=" << avg_pre
              << " infer=" << avg_inf
              << " post=" << avg_post
              << " total=" << avg_tot
              << " | p50=" << p50
              << " p90=" << p90
              << " p99=" << p99
              << " | min=" << min_total
              << " max=" << max_total;
  }
};

static inline void Emit(const EventCallback& cb, const DetectorEvent& ev) {
  if (cb) cb(ev);
}

}  // namespace

struct YoloSegDetector::Impl {
  YoloConfig cfg;
  EventCallback cb;

  std::unique_ptr<OrtBackend> backend;
  RuntimeStatus status;

  bool initialized = false;

  // Perf
  PerfStats perf;

  // Controls:
  // - per_frame_log: print every frame (your requirement)
  // - profile_every: print rolling summary every N frames
  bool per_frame_log = true;  // default true to match your requirement
};

YoloSegDetector::YoloSegDetector(const YoloConfig& cfg)
    : impl_(std::make_unique<Impl>()) {
  impl_->cfg = cfg;
  // If user explicitly disables verbose_log, we still keep per-frame logs off.
  // But your stated requirement is to print per-frame; so default is ON unless verbose_log==false.
  impl_->per_frame_log = cfg.verbose_log;
}

YoloSegDetector::~YoloSegDetector() {
  // Final summary at destruction time (covers "最后的统计")
  if (impl_ && impl_->backend) {
    impl_->perf.LogSummary("FINAL", impl_->backend->provider_name());
  } else if (impl_) {
    impl_->perf.LogSummary("FINAL", impl_->status.active_provider.empty() ? "None" : impl_->status.active_provider);
  }
}

void YoloSegDetector::SetEventCallback(EventCallback cb) {
  impl_->cb = std::move(cb);
}

RuntimeStatus YoloSegDetector::GetStatus() const {
  return impl_->status;
}

bool YoloSegDetector::Init(std::string* err) {
  Emit(impl_->cb, {EventSeverity::kInfo, EventCode::kInitStarted,
                   "Initializing YOLO Seg...", "", "", ""});
  impl_->backend = std::make_unique<OrtBackend>();

  std::string last_err;

  auto try_provider = [&](Provider p) -> bool {
    std::string e;
    if (!impl_->backend->Init(impl_->cfg, p, &e)) {
      last_err = e;
      Emit(impl_->cb, {EventSeverity::kWarning, EventCode::kProviderInitFailed,
                       std::string(ProviderToString(p)) + " init failed.",
                       e, ProviderToString(p), ""});
      LOG(WARNING) << "[YOLO][Init] provider=" << ProviderToString(p)
                   << " init failed: " << e;
      return false;
    }
    Emit(impl_->cb, {EventSeverity::kInfo, EventCode::kUsingProvider,
                     "Using provider: " + impl_->backend->provider_name(),
                     "", "", impl_->backend->provider_name()});
    LOG(INFO) << "[YOLO][Init] Using provider: " << impl_->backend->provider_name()
              << " (preferred=" << ProviderToString(p) << ")";
    return true;
  };

  bool ok = false;
  Provider pref = impl_->cfg.preferred_provider;

  ok = try_provider(pref);

  if (!ok && impl_->cfg.enable_auto_fallback && pref != Provider::kCpu) {
    Emit(impl_->cb, {EventSeverity::kWarning, EventCode::kFallbackToCpu,
                     std::string(ProviderToString(pref)) + " unavailable, falling back to CPU (slower).",
                     last_err, ProviderToString(pref), "CPU"});
    LOG(WARNING) << "[YOLO][Init] Fallback to CPU. from=" << ProviderToString(pref)
                 << " err=" << last_err;

    ok = try_provider(Provider::kCpu);
    impl_->status.is_fallback = ok;
  }

  if (!ok) {
    if (err) *err = last_err.empty() ? "Init failed." : last_err;
    impl_->status.active_provider = "None";
    impl_->status.last_error = last_err;
    Emit(impl_->cb, {EventSeverity::kError, EventCode::kInitFailed,
                     "Init failed.", last_err, "", ""});
    LOG(ERROR) << "[YOLO][Init] Failed: " << last_err;
    impl_->initialized = false;
    return false;
  }

  impl_->status.active_provider = impl_->backend->provider_name();
  impl_->status.last_error.clear();
  impl_->initialized = true;

  Emit(impl_->cb, {EventSeverity::kInfo, EventCode::kInitSucceeded,
                   "YOLO Seg ready.", "", "", ""});

  LOG(INFO) << "[YOLO][Init] Ready. provider=" << impl_->backend->provider_name()
            << " fallback=" << (impl_->status.is_fallback ? 1 : 0)
            << " input=" << impl_->cfg.input_w << "x" << impl_->cfg.input_h
            << " conf=" << impl_->cfg.conf_thresh
            << " iou=" << impl_->cfg.iou_thresh
            << " top_k=" << impl_->cfg.top_k;

  return true;
}

bool YoloSegDetector::Infer(const cv::Mat& bgr, YoloResult* out, std::string* err) {
  if (!impl_->initialized) { if (err) *err = "Call Init() first."; return false; }
  if (!out) { if (err) *err = "out is null."; return false; }

  out->dets.clear();
  out->has_letterbox = false;
  out->has_stats = false;

  if (bgr.empty() || bgr.type() != CV_8UC3) { if (err) *err = "Input must be CV_8UC3 BGR."; return false; }

  InferenceStats stats;
  LetterBoxInfo lb;
  cv::Mat blob;

  const auto t_all0 = std::chrono::high_resolution_clock::now();

  {
    ScopedTimer t(impl_->cfg.enable_profiling ? &stats.preprocess_ms : nullptr);
    blob = LetterBoxBgrToBlobCHW(bgr, impl_->cfg.input_w, impl_->cfg.input_h,
                                 impl_->cfg.keep_aspect_letterbox, &lb);
  }

  std::vector<OrtOutput> outs;
  {
    ScopedTimer t(impl_->cfg.enable_profiling ? &stats.infer_ms : nullptr);
    if (!impl_->backend->Run(blob, &outs, err)) {
      impl_->status.last_error = (err && !err->empty()) ? *err : "infer failed";
      Emit(impl_->cb, {EventSeverity::kError, EventCode::kInferFailed,
                       "Infer failed.", impl_->status.last_error,
                       impl_->backend->provider_name(), ""});
      LOG(ERROR) << "[YOLO][Infer] failed provider=" << impl_->backend->provider_name()
                 << " err=" << impl_->status.last_error;
      return false;
    }
  }

  if (outs.size() < 2) { if (err) *err = "Expected at least 2 outputs (dets, proto)."; return false; }

  OrtOutput det_out = outs[0];
  OrtOutput proto_out = outs[1];
  if (outs[0].shape.size() == 4 && outs[1].shape.size() == 3) { det_out = outs[1]; proto_out = outs[0]; }

  {
    ScopedTimer t(impl_->cfg.enable_profiling ? &stats.postprocess_ms : nullptr);
    if (!PostprocessYoloSeg(impl_->cfg, bgr.size(), lb, det_out, proto_out, out, err)) {
      LOG(ERROR) << "[YOLO][Post] failed err=" << (err ? *err : "");
      return false;
    }
  }

  const auto t_all1 = std::chrono::high_resolution_clock::now();
  stats.total_ms = std::chrono::duration<double, std::milli>(t_all1 - t_all0).count();

  if (impl_->cfg.enable_profiling) { out->has_stats = true; out->stats = stats; }
  out->has_letterbox = true;
  out->letterbox = lb;

  // ---- store perf ----
  impl_->perf.Add(stats);

  // ---- per-frame log (your requirement) ----
  if (impl_->per_frame_log) {
    const uint64_t run_id = impl_->perf.Num();
    LOG(INFO) << "[YOLO][Frame] id=" << run_id
              << " provider=" << impl_->backend->provider_name()
              << " src=" << bgr.cols << "x" << bgr.rows
              << " input=" << impl_->cfg.input_w << "x" << impl_->cfg.input_h
              << " dets=" << out->dets.size()
              << " ms(pre=" << stats.preprocess_ms
              << " infer=" << stats.infer_ms
              << " post=" << stats.postprocess_ms
              << " total=" << stats.total_ms << ")";
  }

  // ---- periodic summary (rolling) ----
  if (impl_->cfg.profile_every > 0 &&
      (impl_->perf.Num() % static_cast<uint64_t>(impl_->cfg.profile_every) == 0)) {
    impl_->perf.LogSummary("ROLLING", impl_->backend->provider_name());
    Emit(impl_->cb, {EventSeverity::kInfo, EventCode::kPerf,
                     "Perf summary logged (rolling).", "",
                     impl_->backend->provider_name(), ""});
  }

  return true;
}

bool YoloSegDetector::InferBatch(const std::vector<cv::Mat>& bgr_list,
                                 std::vector<YoloResult>* out_list,
                                 std::string* err) {
  if (!out_list) { if (err) *err = "out_list is null."; return false; }
  out_list->clear();
  out_list->reserve(bgr_list.size());

  for (const auto& img : bgr_list) {
    YoloResult r;
    std::string e;
    if (!Infer(img, &r, &e)) { if (err) *err = e; return false; }
    out_list->push_back(std::move(r));
  }

  // Batch end summary (optional but usually what you want)
  if (impl_->backend) {
    impl_->perf.LogSummary("BATCH_END", impl_->backend->provider_name());
  }
  return true;
}

}  // namespace yolo
}  // namespace airsteady
