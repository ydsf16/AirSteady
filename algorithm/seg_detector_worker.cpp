#include "seg_detector_worker.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <utility>

#include <glog/logging.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "common/types.h"  // for PreFrame

namespace airsteady {

namespace {

using Clock = std::chrono::steady_clock;

double MsSince(const Clock::time_point& t0, const Clock::time_point& t1) {
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

}  // namespace

SegDetectorWorker::SegDetectorWorker(
    const Config& config,
    std::shared_ptr<yolo::YoloSegDetector> yolo_seg_detector)
    : config_(config), yolo_seg_detector_(std::move(yolo_seg_detector)) {
  CHECK(yolo_seg_detector_ != nullptr)
      << "SegDetectorWorker requires a valid YoloSegDetector.";

  LOG(INFO) << "[SegDetectorWorker] Created. detect_every_n_frames="
            << config_.detect_every_n_frames
            << ", max_num_good_features=" << config_.max_num_good_features
            << ", select_obj_name=" << config_.select_obj_name;

  worker_thread_ = std::thread(&SegDetectorWorker::WorkerLoop, this);
}

SegDetectorWorker::~SegDetectorWorker() {
  Stop();
  LOG(INFO) << "[SegDetectorWorker] Destroyed.";
}

void SegDetectorWorker::Stop() {
  bool expected = false;
  if (!stop_requested_.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel)) {
    // Stop already requested, do nothing.
  }

  {
    std::lock_guard<std::mutex> lock(in_mutex_);
    // Just ensure we hold the mutex when notifying to avoid missed wake-ups.
  }
  in_cv_.notify_all();

  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }
}

void SegDetectorWorker::FeedFrame(const std::shared_ptr<PreFrame>& pre_frame) {
  if (!pre_frame) {
    LOG(WARNING) << "[SegDetectorWorker] FeedFrame: received null PreFrame.";
    return;
  }

  {
    std::lock_guard<std::mutex> lock(in_mutex_);
    if (stop_requested_.load(std::memory_order_acquire)) {
      VLOG(1) << "[SegDetectorWorker] FeedFrame called after Stop(), ignore.";
      return;
    }

    if (pre_frame_buffer_.size() >= kMaxInputBufferSize) {
      // To avoid unbounded memory growth, drop the oldest frame.
      LOG(WARNING) << "[SegDetectorWorker] Input buffer full (size="
                   << pre_frame_buffer_.size()
                   << "), dropping oldest frame.";
      pre_frame_buffer_.pop_front();
    }

    pre_frame_buffer_.push_back(pre_frame);
  }

  in_cv_.notify_one();
}

int SegDetectorWorker::GetProcessedFrameIdx() const {
  return processed_frame_idx_.load(std::memory_order_acquire);
}

bool SegDetectorWorker::QueryResult(int frame_idx, SegDetectorRes* res) const {
  if (!res) {
    return false;
  }

  std::lock_guard<std::mutex> lock(out_mutex_);
  auto it = frame_idx_result_map_.find(frame_idx);
  if (it == frame_idx_result_map_.end()) {
    return false;
  }
  if (!it->second) {
    return false;
  }

  *res = *(it->second);
  return true;
}

void SegDetectorWorker::WorkerLoop() {
  LOG(INFO) << "[SegDetectorWorker] Worker thread started.";

  while (true) {
    std::shared_ptr<PreFrame> pre_frame;

    {
      std::unique_lock<std::mutex> lock(in_mutex_);
      in_cv_.wait(lock, [this]() {
        return stop_requested_.load(std::memory_order_acquire) ||
               !pre_frame_buffer_.empty();
      });

      if (stop_requested_.load(std::memory_order_acquire) &&
          pre_frame_buffer_.empty()) {
        break;
      }

      if (pre_frame_buffer_.empty()) {
        continue;
      }

      pre_frame = pre_frame_buffer_.front();
      pre_frame_buffer_.pop_front();
    }

    if (!pre_frame) {
      continue;
    }

    const int frame_idx = pre_frame->frame_idx;
    processed_frame_idx_.store(frame_idx, std::memory_order_release);

    if (!ShouldRunDetect(frame_idx)) {
      // Skip YOLO on this frame, only update processed_frame_idx_.
      VLOG(1) << "[SegDetectorWorker] Skip detect on frame " << frame_idx;
      continue;
    }

    if (!yolo_seg_detector_->Initialized()) {
      std::string init_err;
      if (!yolo_seg_detector_->Init(&init_err)) {
        LOG(ERROR) << "Failed to init yolo seg!!!";
        break;
      }
    }

    auto res = std::make_shared<SegDetectorRes>();
    res->frame_idx = frame_idx;
    res->time_ns = pre_frame->time_ns;

    RunDetection(pre_frame, res.get());

    {
      std::lock_guard<std::mutex> lock(out_mutex_);
      frame_idx_result_map_[frame_idx] = res;
      result_frame_order_.push_back(frame_idx);
      TrimResultCacheIfNeeded();
    }
  }

  LOG(INFO) << "[SegDetectorWorker] Worker thread exited.";
}

bool SegDetectorWorker::ShouldRunDetect(int frame_idx) {
  if (config_.detect_every_n_frames <= 0) {
    // Fallback: always detect.
    return true;
  }

  if (last_detect_frame_idx_ < 0) {
    return true;
  }

  const int delta = frame_idx - last_detect_frame_idx_;
  return delta >= config_.detect_every_n_frames;
}
void SegDetectorWorker::RunDetection(const std::shared_ptr<PreFrame>& pre_frame,
                                     SegDetectorRes* out_res) {
  if (!out_res) {
    return;
  }

  const int frame_idx = pre_frame->frame_idx;
  auto& timing = out_res->timing;

  if (pre_frame->proxy_bgr.empty()) {
    LOG(WARNING) << "[SegDetectorWorker] Frame " << frame_idx
                 << " has empty proxy_bgr. Skip YOLO.";
    return;
  }

  yolo::YoloResult det_res;
  std::string err;

  const auto t_start = Clock::now();

  // YOLO inference.
  const auto t_yolo_begin = Clock::now();
  bool ok = yolo_seg_detector_->Infer(pre_frame->proxy_bgr, &det_res, &err);
  const auto t_yolo_end = Clock::now();
  timing.yolo_ms = MsSince(t_yolo_begin, t_yolo_end);

  if (!ok) {
    LOG(ERROR) << "[SegDetectorWorker] YOLO failed on frame " << frame_idx
               << ". err=" << err;
    timing.total_ms = MsSince(t_start, Clock::now());
    return;
  }

  last_detect_frame_idx_ = frame_idx;

  // ---------------- 过滤出指定类别的 det，写入 yolo_objects ----------------
  const std::string& target_name = config_.select_obj_name;
  const std::size_t total_dets = det_res.dets.size();

  out_res->yolo_objects.clear();
  out_res->yolo_objects.reserve(total_dets);

  for (const auto& det : det_res.dets) {
    const std::string cls_name = yolo_seg_detector_->ClassName(det.class_id);
    if (cls_name == target_name) {
      out_res->yolo_objects.push_back(det);
    }
  }
  const std::size_t filtered_dets = out_res->yolo_objects.size();

  // Object selection + GFTT（仍然用完整 det_res 做选择逻辑）
  const auto t_select_begin = Clock::now();
  SelectObjectAndExtractGftt(pre_frame->proxy_bgr,
                             pre_frame->proxy_gray,
                             det_res,
                             out_res);
  const auto t_select_end = Clock::now();
  timing.select_object_ms = MsSince(t_select_begin, t_select_end);

  const auto t_end = Clock::now();
  timing.total_ms = MsSince(t_start, t_end);

  if (config_.enable_debug) {
    RenderDebugImage(pre_frame->proxy_bgr, *out_res);
  }

  LOG(INFO) << "[SegDetectorWorker] Frame " << frame_idx
            << " dets_total=" << total_dets
            << ", dets_filtered(" << target_name << ")=" << filtered_dets
            << ", yolo_ms=" << timing.yolo_ms
            << ", select_ms=" << timing.select_object_ms
            << ", gftt_ms=" << timing.gftt_ms
            << ", total_ms=" << timing.total_ms
            << ", has_select_object=" << out_res->has_select_object
            << ", good_features=" << out_res->good_pts_to_track.size();
}

void SegDetectorWorker::SelectObjectAndExtractGftt(
    const cv::Mat& proxy_bgr,
    const cv::Mat& proxy_gray,
    const yolo::YoloResult& det_res,
    SegDetectorRes* out_res) {
  if (!out_res) {
    return;
  }

  auto& timing = out_res->timing;
  const int frame_idx = out_res->frame_idx;

  const auto t_begin = Clock::now();

  // ---------------- 1. Select target object ----------------
  std::vector<const yolo::Det*> candidates;
  candidates.reserve(det_res.dets.size());

  for (const auto& det : det_res.dets) {
    const std::string class_name = yolo_seg_detector_->ClassName(det.class_id);
    if (class_name == config_.select_obj_name) {
      candidates.push_back(&det);
    }
  }

  if (candidates.size() != 1) {
    if (!candidates.empty()) {
      LOG(INFO) << "[SegDetectorWorker] Frame " << frame_idx
                << " has " << candidates.size()
                << " candidates for class=" << config_.select_obj_name
                << ", expect exactly 1. Skip selection.";
    }
    const auto t_end = Clock::now();
    timing.select_object_ms = MsSince(t_begin, t_end);
    return;
  }

  const yolo::Det* selected = candidates.front();
  out_res->has_select_object = true;
  out_res->select_object = *selected;

  // ---------------- 2. Prepare gray image ----------------
  cv::Mat gray;
  if (!proxy_gray.empty() && proxy_gray.channels() == 1 &&
      proxy_gray.size() == proxy_bgr.size()) {
    gray = proxy_gray;
  } else {
    if (proxy_bgr.channels() == 3) {
      cv::cvtColor(proxy_bgr, gray, cv::COLOR_BGR2GRAY);
    } else if (proxy_bgr.channels() == 1) {
      gray = proxy_bgr;
    } else {
      LOG(WARNING) << "[SegDetectorWorker] Frame " << frame_idx
                   << " unexpected channel count="
                   << proxy_bgr.channels() << " for GFTT.";
      const auto t_end = Clock::now();
      timing.select_object_ms = MsSince(t_begin, t_end);
      return;
    }
  }

  // ---------------- 3. ROI: bbox ∩ image ----------------
  cv::Rect img_rect(0, 0, gray.cols, gray.rows);
  cv::Rect roi_box = selected->box & img_rect;
  if (roi_box.empty()) {
    LOG(WARNING) << "[SegDetectorWorker] Frame " << frame_idx
                 << " bbox is empty after clipping. Skip GFTT.";
    const auto t_end = Clock::now();
    timing.select_object_ms = MsSince(t_begin, t_end);
    return;
  }

  cv::Mat gray_roi = gray(roi_box);

  // ---------------- 4. Build mask for ROI ----------------
  cv::Mat gftt_mask;
  if (!selected->mask.empty()) {
    // 根据 PostprocessYoloSeg，mask 是 bbox 尺寸的 ROI。
    if (selected->mask.size() == roi_box.size()) {
      gftt_mask = selected->mask;
    } else {
      // 防御性代码：理论上不会进来。
      cv::resize(selected->mask, gftt_mask, roi_box.size(),
                 0, 0, cv::INTER_NEAREST);
    }
  } else {
    // 没有 seg mask，则全 bbox 区域允许提点。
    gftt_mask = cv::Mat(roi_box.size(), CV_8UC1, cv::Scalar(255));
    LOG(INFO) << "No mask here!!!";
    return;
  }

  // ---------------- 5. GFTT ----------------
  if (config_.max_num_good_features == 0) {
    LOG(INFO) << "[SegDetectorWorker] Frame " << frame_idx
              << " max_num_good_features=0, skip GFTT.";
    const auto t_end = Clock::now();
    timing.select_object_ms = MsSince(t_begin, t_end);
    return;
  }

  std::vector<cv::Point2f> pts_local;
  const int max_corners = static_cast<int>(std::min<std::size_t>(
      config_.max_num_good_features,
      static_cast<std::size_t>(std::numeric_limits<int>::max())));

  const auto t_gftt_begin = Clock::now();
  cv::goodFeaturesToTrack(
      gray_roi, 
      pts_local, 
      max_corners,
      /*qualityLevel=*/0.001,
      /*minDistance=*/2.0,
      /*mask=*/gftt_mask);

  const auto t_gftt_end = Clock::now();
  timing.gftt_ms = MsSince(t_gftt_begin, t_gftt_end);

  // ---------------- 6. Convert to full-image coordinates ----------------
  out_res->good_pts_to_track.clear();
  out_res->good_pts_to_track.reserve(pts_local.size());
  for (const auto& p : pts_local) {
    out_res->good_pts_to_track.emplace_back(
        p.x + static_cast<float>(roi_box.x),
        p.y + static_cast<float>(roi_box.y));
  }

  const auto t_end = Clock::now();
  timing.select_object_ms = MsSince(t_begin, t_end);

  LOG(INFO) << "[SegDetectorWorker] Frame " << frame_idx
            << " selected object (class=" << config_.select_obj_name
            << "), good_features=" << out_res->good_pts_to_track.size()
            << ", gftt_ms=" << timing.gftt_ms
            << ", select_total_ms=" << timing.select_object_ms;
}

void SegDetectorWorker::TrimResultCacheIfNeeded() {
  if (frame_idx_result_map_.size() <= kMaxResultCacheSize) {
    return;
  }

  while (frame_idx_result_map_.size() > kMaxResultCacheSize &&
         !result_frame_order_.empty()) {
    const int oldest_frame_idx = result_frame_order_.front();
    result_frame_order_.pop_front();

    auto it = frame_idx_result_map_.find(oldest_frame_idx);
    if (it != frame_idx_result_map_.end()) {
      frame_idx_result_map_.erase(it);
    }
  }

  LOG(INFO) << "[SegDetectorWorker] Trimmed result cache to size="
            << frame_idx_result_map_.size();
}

void SegDetectorWorker::RenderDebugImage(const cv::Mat& proxy_bgr,
                                         const SegDetectorRes& res) {
  if (proxy_bgr.empty()) {
    return;
  }

  cv::Mat debug_img;
  if (proxy_bgr.channels() == 3) {
    debug_img = proxy_bgr.clone();
  } else if (proxy_bgr.channels() == 1) {
    cv::cvtColor(proxy_bgr, debug_img, cv::COLOR_GRAY2BGR);
  } else {
    LOG(WARNING) << "[SegDetectorWorker] RenderDebugImage: unexpected channels="
                 << proxy_bgr.channels();
    return;
  }

  const cv::Scalar color_all_box(0, 255, 255);     // 黄：所有检测框
  const cv::Scalar color_selected_box(0, 255, 0);  // 绿：选中的目标
  const cv::Scalar color_gftt_pt(0, 0, 255);       // 红：GFTT 点

  const cv::Rect img_rect(0, 0, debug_img.cols, debug_img.rows);

  // 1) 画所有检测框（细线）
  for (const auto& det : res.yolo_objects) {
    cv::Rect box = det.box & img_rect;
    if (box.empty()) {
      continue;
    }
    cv::rectangle(debug_img, box, color_all_box, 1);

    // 在左上角写类别和分数
    std::string label = yolo_seg_detector_->ClassName(det.class_id) + " " +
                        cv::format("%.2f", det.score);
    int base_line = 0;
    cv::Size label_size =
        cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &base_line);
    cv::Rect bg_rect(box.x,
                     std::max(box.y - label_size.height, 0),
                     label_size.width + 2,
                     label_size.height + base_line + 2);
    bg_rect &= img_rect;
    if (!bg_rect.empty()) {
      cv::rectangle(debug_img, bg_rect, cv::Scalar(0, 0, 0), cv::FILLED);
      cv::putText(debug_img, label,
                  cv::Point(bg_rect.x + 1, bg_rect.y + label_size.height),
                  cv::FONT_HERSHEY_SIMPLEX, 0.4,
                  cv::Scalar(255, 255, 255), 1);
    }
  }

  // 2) 高亮选中的目标（粗线 + mask overlay）
  if (res.has_select_object) {
    cv::Rect sel_box = res.select_object.box & img_rect;
    if (!sel_box.empty()) {
      cv::rectangle(debug_img, sel_box, color_selected_box, 2);
    }

    // 如果有 mask，就叠加一个半透明 overlay
    if (!res.select_object.mask.empty() && !sel_box.empty()) {
      cv::Mat overlay = debug_img.clone();

      cv::Rect roi_box = sel_box;  // ROI in image coords
      cv::Mat ov_roi = overlay(roi_box);

      cv::Mat mask = res.select_object.mask;
      if (mask.size() != roi_box.size()) {
        cv::resize(mask, mask, roi_box.size(), 0, 0, cv::INTER_NEAREST);
      }

      // 在 mask==255 的地方涂上颜色（例如蓝色）
      ov_roi.setTo(cv::Scalar(255, 0, 0), mask);

      // blend overlay -> debug_img
      const double alpha = 0.4;
      cv::addWeighted(overlay, alpha, debug_img, 1.0 - alpha, 0.0, debug_img);
    }
  }

  // 3) 画 GFTT 点
  for (const auto& pt : res.good_pts_to_track) {
    cv::Point ipt(cvRound(pt.x), cvRound(pt.y));
    if (!img_rect.contains(ipt)) {
      continue;
    }
    cv::circle(debug_img, ipt, 2, color_gftt_pt, cv::FILLED);
  }

  // 4) 左上角写一行调试文字：frame / dets / timing
  const std::string info = cv::format(
      "f=%d dets=%zu yolo=%.1fms gftt=%.1fms sel=%.1fms tot=%.1fms",
      res.frame_idx,
      res.yolo_objects.size(),
      res.timing.yolo_ms,
      res.timing.gftt_ms,
      res.timing.select_object_ms,
      res.timing.total_ms);

  cv::putText(debug_img, info, cv::Point(5, 18),
              cv::FONT_HERSHEY_SIMPLEX, 0.5,
              cv::Scalar(0, 255, 0), 1);

  // 5) 显示窗口（仅调试用）
  const std::string win_name = config_.debug_window_name.empty()
                                   ? "SegDetectorDebug"
                                   : config_.debug_window_name;

  cv::imshow(win_name, debug_img);
  // 为了不卡住线程，用 1ms 非阻塞
  cv::waitKey(1);
}

}  // namespace airsteady
