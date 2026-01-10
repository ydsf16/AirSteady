#include "algorithm/tracker.h"

#include <algorithm>
#include <chrono>
#include <deque>
#include <iomanip>
#include <sstream>
#include <thread>

#include <glog/logging.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include "common/file_utils.hpp"

namespace airsteady {

namespace {
using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

double MsSince(const Clock::time_point& t0, const Clock::time_point& t1) {
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

std::string ZeroPadInt(int v, int width = 6) {
  std::ostringstream oss;
  oss << std::setw(width) << std::setfill('0') << v;
  return oss.str();
}

void DrawPoints(cv::Mat& bgr,
                const std::vector<cv::Point2f>& pts,
                int radius,
                const cv::Scalar& color) {
  if (bgr.empty()) return;
  for (const auto& p : pts) {
    cv::circle(bgr, p, radius, color, /*thickness=*/-1, cv::LINE_AA);
  }
}

void DrawFlowPairs(cv::Mat& bgr,
                   const std::vector<cv::Point2f>& prev_pts,
                   const std::vector<cv::Point2f>& curr_pts,
                   const std::vector<int>& inlier_indices,
                   bool draw_outliers) {
  if (bgr.empty()) return;

  // inlier_indices are indices into prev_pts/curr_pts (same indexing).
  const auto is_inlier = [&](int i) -> bool {
    for (int idx : inlier_indices) {
      if (idx == i) return true;
    }
    return false;
  };

  const int n = static_cast<int>(std::min(prev_pts.size(), curr_pts.size()));
  for (int i = 0; i < n; ++i) {
    const bool inl = is_inlier(i);
    if (!inl && !draw_outliers) continue;

    // Green=inlier, Red=outlier.
    const cv::Scalar c = inl ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
    cv::arrowedLine(bgr, prev_pts[i], curr_pts[i], c, 1, cv::LINE_AA, 0, 0.25);
    cv::circle(bgr, curr_pts[i], 2, c, -1, cv::LINE_AA);
  }
}

cv::Mat MakeSideBySide(const cv::Mat& left_bgr,
                       const cv::Mat& right_bgr,
                       const std::string& left_title,
                       const std::string& right_title) {
  if (left_bgr.empty() || right_bgr.empty()) return cv::Mat();

  cv::Mat L = left_bgr.clone();
  cv::Mat R = right_bgr.clone();

  if (L.type() != CV_8UC3) {
    if (L.channels() == 1) {
      cv::cvtColor(L, L, cv::COLOR_GRAY2BGR);
    } else {
      L.convertTo(L, CV_8UC3);
    }
  }
  if (R.type() != CV_8UC3) {
    if (R.channels() == 1) {
      cv::cvtColor(R, R, cv::COLOR_GRAY2BGR);
    } else {
      R.convertTo(R, CV_8UC3);
    }
  }

  // Make same height.
  if (L.rows != R.rows) {
    const int h = std::min(L.rows, R.rows);
    cv::resize(L, L, cv::Size(static_cast<int>(1.0 * L.cols * h / L.rows), h));
    cv::resize(R, R, cv::Size(static_cast<int>(1.0 * R.cols * h / R.rows), h));
  }

  cv::Mat out(L.rows, L.cols + R.cols, CV_8UC3);
  L.copyTo(out(cv::Rect(0, 0, L.cols, L.rows)));
  R.copyTo(out(cv::Rect(L.cols, 0, R.cols, R.rows)));

  const int font = cv::FONT_HERSHEY_SIMPLEX;
  cv::putText(out, left_title, cv::Point(10, 25), font, 0.7,
              cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
  cv::putText(out, right_title, cv::Point(L.cols + 10, 25), font, 0.7,
              cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
  return out;
}

}  // namespace

Tracker::Tracker(VideoPreprocessor* video_preprocessor, const Config& config)
    : video_preprocessor_(video_preprocessor), config_(config) {
  CHECK(video_preprocessor_ != nullptr);

  // ---------------- YOLO init ----------------
  yolo::YoloConfig yolo_cfg;
  std::string exe_folder;
  GetExeFolder(&exe_folder);
  yolo_cfg.onnx_path = exe_folder + "/model/model.onnx";

  LOG(INFO) << "[Tracker] YOLO onnx_path: " << yolo_cfg.onnx_path;

  yolo_cfg.preferred_provider = airsteady::yolo::Provider::kDirectML;
  yolo_cfg.enable_auto_fallback = true;
  yolo_cfg.verbose_log = true;
  yolo_cfg.enable_profiling = true;

  yolo_seg_detector_ = std::make_shared<yolo::YoloSegDetector>(yolo_cfg);

  yolo_seg_detector_->SetEventCallback(
      [](const airsteady::yolo::DetectorEvent& ev) {
        LOG(INFO) << "[YoloSegDetector] sev=" << static_cast<int>(ev.severity)
                  << ", code=" << static_cast<int>(ev.code)
                  << ", msg=" << ev.message << ", from=" << ev.from_provider
                  << ", to=" << ev.to_provider << ", details=" << ev.details;
      });

  // ---------------- SegDetectorWorker init ----------------
  SegDetectorWorker::Config seg_detect_worker_cfg;
  seg_detect_worker_cfg.select_obj_name = config_.select_obj_name;  // COCO airplane
  seg_detect_worker_cfg.detect_every_n_frames = config_.yolo_detect_every_n_frames;
  seg_detect_worker_cfg.max_num_good_features = config_.max_num_good_features;
  seg_detect_worker_ =
      std::make_shared<SegDetectorWorker>(seg_detect_worker_cfg, yolo_seg_detector_);
}

Tracker::~Tracker() {
  StopTracking();
  if (seg_detect_worker_) {
    seg_detect_worker_->Stop();
  }
}

bool Tracker::StartTracking() {
  if (thread_ && thread_->joinable()) {
    LOG(WARNING) << "[Tracker] StartTracking called while already running.";
    return false;
  }

  stop_.store(false, std::memory_order_relaxed);
  thread_ = std::make_shared<std::thread>(&Tracker::Run, this);
  return true;
}

void Tracker::StopTracking() {
  stop_.store(true, std::memory_order_relaxed);
  if (thread_ && thread_->joinable()) {
    thread_->join();
  }
}

void Tracker::AddTrackingResultCallback(TrackingResultCallback cb) {
  tracking_result_cbs_.push_back(std::move(cb));
}

void Tracker::AddTrackFinishedCallback(TrackFinishedCallback cb) {
  track_finished_cbs_.push_back(std::move(cb));
}

std::vector<FrameTrackingResult> Tracker::GetTrackingResults() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return track_results_;
}

// 简单的 Translation RANSAC：
// - 模型是 delta = curr - prev
// - 每个点都可以产生一个候选 delta_i
// - 使用 L2 距离阈值 ransac_inlier_thresh 作为 inlier 判定
// - 返回 best 模型的平均 delta + 噪声（平均残差平方）
bool Tracker::EstimateTranslationRansac(
    const std::vector<cv::Point2f>& prev_pts,
    const std::vector<cv::Point2f>& curr_pts,
    Eigen::Vector2d* delta,
    double* noise,
    std::vector<int>* inlier_indices) const {
  if (!delta || !noise || !inlier_indices) {
    return false;
  }
  const int n = static_cast<int>(prev_pts.size());
  if (n <= 0 || static_cast<int>(curr_pts.size()) != n) {
    return false;
  }

  const double thresh_sq =
      config_.ransac_inlier_thresh * config_.ransac_inlier_thresh;

  int best_inlier_count = 0;
  std::vector<int> best_inliers;

  // O(N^2) 简单 RANSAC，N <= 100 级别足够快。
  for (int i = 0; i < n; ++i) {
    const double dx_i = static_cast<double>(curr_pts[i].x - prev_pts[i].x);
    const double dy_i = static_cast<double>(curr_pts[i].y - prev_pts[i].y);

    int inlier_count = 0;
    std::vector<int> inliers;
    inliers.reserve(n);

    for (int j = 0; j < n; ++j) {
      const double dx_j = static_cast<double>(curr_pts[j].x - prev_pts[j].x);
      const double dy_j = static_cast<double>(curr_pts[j].y - prev_pts[j].y);

      const double rx = dx_j - dx_i;
      const double ry = dy_j - dy_i;
      const double err_sq = rx * rx + ry * ry;

      if (err_sq <= thresh_sq) {
        ++inlier_count;
        inliers.push_back(j);
      }
    }

    if (inlier_count > best_inlier_count) {
      best_inlier_count = inlier_count;
      best_inliers.swap(inliers);
    }
  }

  if (best_inlier_count < config_.min_pts_for_ransac) {
    LOG(INFO) << "[Tracker] RANSAC: not enough inliers, best_inlier_count="
              << best_inlier_count
              << ", min_pts_for_ransac=" << config_.min_pts_for_ransac;
    return false;
  }

  const double inlier_ratio =
      static_cast<double>(best_inlier_count) / static_cast<double>(n);
  if (inlier_ratio < config_.ransac_min_inlier_ratio) {
    LOG(INFO) << "[Tracker] RANSAC: inlier_ratio too low, ratio=" << inlier_ratio
              << ", min_ratio=" << config_.ransac_min_inlier_ratio;
    return false;
  }

  // 用最佳 inlier 集合重新算平均 delta 和噪声
  double sum_dx = 0.0;
  double sum_dy = 0.0;
  for (int idx : best_inliers) {
    sum_dx += static_cast<double>(curr_pts[idx].x - prev_pts[idx].x);
    sum_dy += static_cast<double>(curr_pts[idx].y - prev_pts[idx].y);
  }
  const double mean_dx = sum_dx / best_inlier_count;
  const double mean_dy = sum_dy / best_inlier_count;

  double var = 0.0;
  for (int idx : best_inliers) {
    const double dx =
        static_cast<double>(curr_pts[idx].x - prev_pts[idx].x) - mean_dx;
    const double dy =
        static_cast<double>(curr_pts[idx].y - prev_pts[idx].y) - mean_dy;
    var += dx * dx + dy * dy;
  }
  var /= std::max(best_inlier_count, 1);

  *delta = Eigen::Vector2d(mean_dx, mean_dy);
  *noise = var;
  *inlier_indices = best_inliers;

  return true;
}

void Tracker::Run() {
  LOG(INFO) << "[Tracker] Run() started.";

  // 局部状态：缓冲帧 + KLT 前一帧
  std::deque<std::shared_ptr<PreFrame>> buffer_frames;
  std::shared_ptr<PreFrame> prev_frame;
  std::vector<cv::Point2f> prev_pts;

  // Debug draw config (requires you to provide these in Config if you want them dynamic):
  //   - config_.debug (bool)
  //   - config_.debug_draw_every_n (int)  [optional; fallback below]
  //   - config_.debug_draw_dir (std::string) [optional; fallback below]
  const bool debug_draw = false;  // reuse your existing debug flag
  const int debug_every_n =
      (config_.debug_draw_every_n > 0) ? config_.debug_draw_every_n : 1;
  const std::string debug_dir =
      (!config_.debug_draw_dir.empty()) ? config_.debug_draw_dir : "debug_track";

  if (debug_draw) {
    CreateFolder(debug_dir);
    LOG(INFO) << "[Tracker][DebugDraw] enabled, dir=" << debug_dir
              << ", every_n=" << debug_every_n;
  }

  // 对单帧的处理逻辑封装成 lambda，结束时也可复用
  auto process_one_frame = [&](const std::shared_ptr<PreFrame>& curr_frame) {
    if (!curr_frame || stop_.load(std::memory_order_relaxed)) {
      return;
    }

    const auto t_frame_start = Clock::now();
    double wait_yolo_ms = 0.0;
    double klt_ms = 0.0;
    double ransac_ms = 0.0;

    const int frame_idx = curr_frame->frame_idx;

    // ---------------- 等 seg_detector 处理到当前帧 ----------------
    const auto t_wait_begin = Clock::now();
    while (!stop_.load(std::memory_order_relaxed) &&
           seg_detect_worker_->GetProcessedFrameIdx() < frame_idx) {
      std::this_thread::sleep_for(1ms);
    }
    const auto t_wait_end = Clock::now();
    wait_yolo_ms = MsSince(t_wait_begin, t_wait_end);

    if (stop_.load(std::memory_order_relaxed)) {
      return;
    }

    FrameTrackingResult track_res;
    track_res.frame_idx = curr_frame->frame_idx;
    track_res.time_ns = curr_frame->time_ns;

    std::vector<cv::Point2f> curr_pts;

    // -------- Debug stash for drawing (good points + inliers) --------
    std::vector<cv::Point2f> dbg_good_prev;
    std::vector<cv::Point2f> dbg_good_curr;
    std::vector<int> dbg_inliers;  // indices into dbg_good_prev/curr
    bool dbg_has_klt = false;
    bool dbg_has_ransac = false;

    // ---------------- 1) KLT + RANSAC 平移估计 ----------------
    if (prev_frame && !prev_frame->proxy_gray.empty() &&
        !curr_frame->proxy_gray.empty() && !prev_pts.empty()) {
      const auto t_klt_begin = Clock::now();

      std::vector<cv::Point2f> next_pts;
      std::vector<unsigned char> status;
      std::vector<float> err;

      const cv::Size win_size(21, 21);
      const int max_level = 4;
      const cv::TermCriteria criteria(
          cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 50, 1e-4);

      cv::calcOpticalFlowPyrLK(prev_frame->proxy_gray, curr_frame->proxy_gray,
                               prev_pts, next_pts, status, err, win_size,
                               max_level, criteria);

      std::vector<cv::Point2f> good_prev;
      std::vector<cv::Point2f> good_curr;
      good_prev.reserve(next_pts.size());
      good_curr.reserve(next_pts.size());

      for (std::size_t i = 0; i < next_pts.size(); ++i) {
        if (!status[i]) {
          continue;
        }
        const cv::Point2f& p0 = prev_pts[i];
        const cv::Point2f& p1 = next_pts[i];

        if (p1.x < 0 || p1.y < 0 || p1.x >= curr_frame->proxy_gray.cols ||
            p1.y >= curr_frame->proxy_gray.rows) {
          continue;
        }

        good_prev.push_back(p0);
        good_curr.push_back(p1);
      }

      const auto t_klt_end = Clock::now();
      klt_ms = MsSince(t_klt_begin, t_klt_end);

      dbg_good_prev = good_prev;
      dbg_good_curr = good_curr;
      dbg_has_klt = true;

      if (static_cast<int>(good_curr.size()) >= config_.min_pts_for_ransac) {
        const auto t_ransac_begin = Clock::now();

        Eigen::Vector2d delta;
        double noise = 0.0;
        std::vector<int> inliers;
        const bool ok =
            EstimateTranslationRansac(good_prev, good_curr, &delta, &noise,
                                      &inliers);

        const auto t_ransac_end = Clock::now();
        ransac_ms = MsSince(t_ransac_begin, t_ransac_end);

        dbg_inliers = inliers;
        dbg_has_ransac = ok;

        if (ok) {
          track_res.delta_valid = true;
          track_res.delta = delta;
          track_res.delta_noise = noise;

          // 以 inlier 的 curr 点作为新的 tracked points + global center
          curr_pts.reserve(inliers.size());
          for (int idx : inliers) {
            const auto& pt = good_curr[static_cast<std::size_t>(idx)];
            curr_pts.push_back(pt);
          }

          LOG(INFO) << "[Tracker] Frame " << frame_idx
                    << " KLT+RANSAC ok, inliers=" << curr_pts.size()
                    << ", delta=(" << delta.x() << "," << delta.y()
                    << "), noise=" << noise;
        } else {
          LOG(INFO) << "[Tracker] Frame " << frame_idx
                    << " KLT+RANSAC failed. good_pts=" << good_curr.size();
          curr_pts.clear();
        }
      } else {
        LOG(INFO) << "[Tracker] Frame " << frame_idx
                  << " too few good KLT points: " << good_curr.size() << " < "
                  << config_.min_pts_for_ransac;
        curr_pts.clear();
      }
    }

    // ---------------- 2) YOLO 结果：初始化 / 纠错重置 ----------------
    SegDetectorRes seg_res;
    const bool has_seg = seg_detect_worker_->QueryResult(frame_idx, &seg_res);

    if (has_seg) {
      track_res.seg_detect_res = seg_res;
    }

    track_res.global_center_valid = false;

    if (has_seg && seg_res.has_select_object) {
      const auto& box = seg_res.select_object.box;
      const double bbox_cx =
          static_cast<double>(box.x) + 0.5 * static_cast<double>(box.width);
      const double bbox_cy =
          static_cast<double>(box.y) + 0.5 * static_cast<double>(box.height);

      track_res.global_center_valid = true;  // (per your note: intentional)
      track_res.global_center = Eigen::Vector2d(bbox_cx, bbox_cy);
      LOG(INFO) << "GLOBAL CENTER: " << bbox_cx << ", " << bbox_cy;

      // 2.1 如果没有任何跟踪点，直接用 YOLO good_features 初始化
      if (curr_pts.empty()) {
        curr_pts = seg_res.good_pts_to_track;

        // 首帧用 YOLO 初始化，不信这一帧 delta
        track_res.delta_valid = false;

        LOG(INFO) << "[Tracker] Frame " << frame_idx
                  << " init KLT from YOLO, pts=" << curr_pts.size()
                  << ", center=(" << bbox_cx << "," << bbox_cy << ")";
      } else {
        // 2.2 已有 KLT 跟踪，但点太少 or delta_invalid，尝试用 YOLO 重置
        if (static_cast<int>(curr_pts.size()) < config_.reinit_pts_threshold ||
            !track_res.delta_valid) {
          if (!seg_res.good_pts_to_track.empty()) {
            curr_pts = seg_res.good_pts_to_track;

            track_res.global_center_valid = true;
            track_res.global_center = Eigen::Vector2d(bbox_cx, bbox_cy);

            LOG(INFO) << "[Tracker] Frame " << frame_idx
                      << " re-init KLT from YOLO, pts=" << curr_pts.size()
                      << ", center=(" << bbox_cx << "," << bbox_cy << ")";
          }
        }
      }
    }

    // ---------------- Debug draw: plot tracked points on image ----------------
    if (debug_draw && (frame_idx % std::max(1, debug_every_n) == 0)) {
      // Left: prev frame + prev good points (blue)
      // Right: curr frame + curr good points (blue) + flow arrows (green/red)
      cv::Mat prev_vis;
      if (prev_frame && !prev_frame->proxy_bgr.empty()) {
        prev_vis = prev_frame->proxy_bgr.clone();
        if (prev_vis.type() != CV_8UC3 && prev_vis.channels() == 1) {
          cv::cvtColor(prev_vis, prev_vis, cv::COLOR_GRAY2BGR);
        }
      }
      cv::Mat curr_vis = curr_frame->proxy_bgr.clone();
      if (curr_vis.type() != CV_8UC3 && curr_vis.channels() == 1) {
        cv::cvtColor(curr_vis, curr_vis, cv::COLOR_GRAY2BGR);
      }

      // Draw good points (KLT filtered) if available; otherwise draw the final curr_pts.
      if (!prev_vis.empty()) {
        if (dbg_has_klt) {
          DrawPoints(prev_vis, dbg_good_prev, 2, cv::Scalar(255, 0, 0));  // blue
        }
      }

      if (!curr_vis.empty()) {
        if (dbg_has_klt) {
          DrawPoints(curr_vis, dbg_good_curr, 2, cv::Scalar(255, 0, 0));  // blue
          if (dbg_has_ransac) {
            DrawFlowPairs(curr_vis, dbg_good_prev, dbg_good_curr, dbg_inliers,
                          /*draw_outliers=*/true);
          }
        } else {
          // fallback: only final tracked points
          DrawPoints(curr_vis, curr_pts, 2, cv::Scalar(255, 0, 0));
        }

        // Optionally draw YOLO bbox center if present (yellow cross)
        if (has_seg && seg_res.has_select_object) {
          const auto& box = seg_res.select_object.box;
          const int cx = static_cast<int>(
              static_cast<double>(box.x) + 0.5 * static_cast<double>(box.width));
          const int cy = static_cast<int>(static_cast<double>(box.y) +
                                          0.5 * static_cast<double>(box.height));
          cv::drawMarker(curr_vis, cv::Point(cx, cy), cv::Scalar(0, 255, 255),
                         cv::MARKER_CROSS, 18, 2, cv::LINE_AA);
        }
      }

      cv::Mat out;
      if (!prev_vis.empty()) {
        out = MakeSideBySide(
            prev_vis, curr_vis,
            "prev (blue=good_prev)",
            dbg_has_ransac
                ? "curr (blue=good_curr, green=inlier, red=outlier)"
                : "curr (blue=good_curr)");
      } else {
        out = curr_vis;
      }

      if (!out.empty()) {
        const std::string path =
            debug_dir + "/" + std::to_string(frame_idx) + ".png";
        const bool ok = cv::imwrite(path, curr_frame->proxy_bgr);
        if (!ok) {
          LOG(WARNING) << "[Tracker][DebugDraw] imwrite failed: " << path;
        } else {
          LOG(INFO) << "[Tracker][DebugDraw] saved: " << path;
        }

        if (config_.debug_draw_show) {
          cv::imshow("TrackerDebugDraw", out);
          cv::waitKey(1);
        }
      }
    }

    // ---------------- 3) 保存结果 + 回调预览 ----------------
    {
      std::lock_guard<std::mutex> lock(mutex_);
      track_results_.push_back(track_res);
    }

    if (has_seg) {
      FrameTrackingResultPreview preview;
      preview.seg_detect_res = has_seg ? seg_res : SegDetectorRes{};
      preview.proxy_bgr = curr_frame->proxy_bgr;
      preview.frame_idx = curr_frame->frame_idx;
      preview.time_ns = curr_frame->time_ns;

      LOG(INFO) << "CURR FRAME INDEX " << preview.frame_idx;
      for (const auto& cb : tracking_result_cbs_) {
        cb(preview);
      }
    }

    // 更新 prev_* 状态，准备下一帧
    prev_frame = curr_frame;
    prev_pts = std::move(curr_pts);

    // ---------------- Debug: accumulate delta and write translated video ----------------
    // Purpose: quick visual sanity check for delta integration (NOT production stabilizer).
    // Mechanism: accum += delta (when valid), and translate current frame by -accum.
    {
      // Toggle as you like (keep it hard-coded for debugging).
      const bool debug_accum_stab_video = true;

      if (debug_accum_stab_video && !curr_frame->proxy_bgr.empty()) {
        // Static states for one run (simple debugging, not thread-safe across multiple trackers).
        static bool s_inited = false;
        static cv::VideoWriter s_writer;
        static Eigen::Vector2d s_accum_delta(0.0, 0.0);
        static int s_last_frame_idx = -1;
        static int s_written = 0;

        // Reset when a new run starts or frame index goes backwards.
        if (!s_inited || (s_last_frame_idx >= 0 && frame_idx <= s_last_frame_idx)) {
          s_inited = true;
          s_accum_delta.setZero();
          s_last_frame_idx = -1;
          s_written = 0;

          if (s_writer.isOpened()) {
            s_writer.release();
          }

          // Output path
          const std::string out_path = debug_dir + "/accum_translate_debug.mp4";

          // FPS: best-effort. If you have a better source, replace it.
          // (OpenCV VideoWriter needs a stable fps.)
          double fps = 30.0;
          if (video_preprocessor_) {
            const VideoInfo info = video_preprocessor_->GetVideoInfo();
            if (info.fps > 1e-3) fps = info.fps;
          }

          // FourCC: mp4v is usually available on Windows/macOS with OpenCV builds.
          const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
          const cv::Size sz(curr_frame->proxy_bgr.cols, curr_frame->proxy_bgr.rows);

          const bool ok = s_writer.open(out_path, fourcc, fps, sz, /*isColor=*/true);
          if (!ok) {
            LOG(WARNING) << "[Tracker][AccumStab] Failed to open VideoWriter: " << out_path
                         << " fps=" << fps << " size=" << sz.width << "x" << sz.height;
          } else {
            LOG(INFO) << "[Tracker][AccumStab] Writing debug video: " << out_path
                      << " fps=" << fps << " size=" << sz.width << "x" << sz.height;
          }
        }

        // Accumulate translation if current delta is valid.
        if (track_res.delta_valid) {
          s_accum_delta += track_res.delta;
        }

        // Apply translation by -accum (cancel motion).
        // Note: OpenCV warpAffine uses float/double. Keep double for accuracy.
        cv::Mat stabilized;
        const double tx = -s_accum_delta.x();
        const double ty = -s_accum_delta.y();
        cv::Mat M = (cv::Mat_<double>(2, 3) << 1.0, 0.0, tx,
                                              0.0, 1.0, ty);

        // Border filled with black; you can change to replicate if you prefer.
        cv::warpAffine(curr_frame->proxy_bgr, stabilized, M,
                       curr_frame->proxy_bgr.size(),
                       cv::INTER_LINEAR,
                       cv::BORDER_CONSTANT,
                       cv::Scalar(0, 0, 0));

        // Overlay some text for debugging (frame idx + accum).
        {
          std::ostringstream oss;
          oss << "idx=" << frame_idx
              << " accum=(" << std::fixed << std::setprecision(2)
              << s_accum_delta.x() << "," << s_accum_delta.y() << ")"
              << " delta_valid=" << (track_res.delta_valid ? 1 : 0);
          cv::putText(stabilized, oss.str(), cv::Point(10, 30),
                      cv::FONT_HERSHEY_SIMPLEX, 0.7,
                      cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        }

        if (s_writer.isOpened()) {
          // Ensure size matches writer settings.
          if (stabilized.cols == static_cast<int>(s_writer.get(cv::VIDEOWRITER_PROP_FRAMEBYTES)) /*not reliable*/ ) {
            // ignore; property is not consistent across backends
          }
          s_writer.write(stabilized);
          ++s_written;

          if ((s_written % 30) == 0) {
            LOG(INFO) << "[Tracker][AccumStab] wrote_frames=" << s_written
                      << " last_idx=" << frame_idx
                      << " accum=(" << std::fixed << std::setprecision(2)
                      << s_accum_delta.x() << "," << s_accum_delta.y() << ")";
          }
        }

        s_last_frame_idx = frame_idx;
      }
    }

    // ---------------- 4) 统计总耗时并打印 ----------------
    const auto t_frame_end = Clock::now();
    const double total_ms = MsSince(t_frame_start, t_frame_end);

    LOG(INFO) << "[Tracker] Frame " << frame_idx
              << " timing_ms { wait_yolo=" << wait_yolo_ms
              << ", klt=" << klt_ms << ", ransac=" << ransac_ms
              << ", total=" << total_ms << " }"
              << ", delta_valid=" << track_res.delta_valid
              << ", delta=(" << track_res.delta.x() << "," << track_res.delta.y()
              << ")"
              << ", pts_prev=" << prev_pts.size() << ", has_seg=" << has_seg
              << ", has_select_obj=" << (has_seg && seg_res.has_select_object);
  };

  // 主循环：不断取帧、喂 seg_detector，并按 delay 处理
  while (!stop_.load(std::memory_order_relaxed)) {
    std::shared_ptr<PreFrame> frame = video_preprocessor_->NextFrame();
    if (!frame) {
      LOG(INFO) << "[Tracker] No more frames from VideoPreprocessor.";
      break;
    }

    // 喂给 YOLO 分割线程
    seg_detect_worker_->FeedFrame(frame);

    // 延迟 N 帧，给 YOLO 一点时间
    buffer_frames.push_back(frame);
    if (static_cast<int>(buffer_frames.size()) <= config_.delay_n_frames) {
      continue;
    }

    auto curr_frame = buffer_frames.front();
    buffer_frames.pop_front();
    process_one_frame(curr_frame);
  }

  // 把缓冲中剩余的帧也处理完
  while (!stop_.load(std::memory_order_relaxed) && !buffer_frames.empty()) {
    auto curr_frame = buffer_frames.front();
    buffer_frames.pop_front();
    process_one_frame(curr_frame);
  }

  // 通知结束
  for (const auto& cb : track_finished_cbs_) {
    cb();
  }

  LOG(INFO) << "[Tracker] Run() finished.";
}

}  // namespace airsteady
