#include "algorithm/tracker.h"

#include <algorithm>
#include <deque>
#include <thread>
#include <chrono>

#include <glog/logging.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include "common/file_utils.hpp"

namespace airsteady {

namespace {
using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

double MsSince(const Clock::time_point& t0,
               const Clock::time_point& t1) {
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}
}  // namespace

Tracker::Tracker(VideoPreprocessor* video_preprocessor,
                 const Config& config)
    : video_preprocessor_(video_preprocessor),
      config_(config) {
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
                  << ", msg=" << ev.message
                  << ", from=" << ev.from_provider
                  << ", to=" << ev.to_provider
                  << ", details=" << ev.details;
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
bool Tracker::EstimateTranslationRansac(const std::vector<cv::Point2f>& prev_pts,
                                        const std::vector<cv::Point2f>& curr_pts,
                                        Eigen::Vector2d* delta,
                                        double* noise,
                                        std::vector<int>* inlier_indices) const {
  if (!delta || !noise || !inlier_indices) {
    return false;
  }
  const int n = static_cast<int>(prev_pts.size());
  if (n <= 0 || curr_pts.size() != prev_pts.size()) {
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
    LOG(INFO) << "[Tracker] RANSAC: inlier_ratio too low, ratio="
              << inlier_ratio
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

  // 对单帧的处理逻辑封装成 lambda，结束时也可复用
  auto process_one_frame =
      [&](const std::shared_ptr<PreFrame>& curr_frame) {
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
        track_res.frame_idx = frame_idx;
        track_res.time_ns = curr_frame->time_ns;

        std::vector<cv::Point2f> curr_pts;

        // ---------------- 1) KLT + RANSAC 平移估计 ----------------
        if (prev_frame &&
            !prev_frame->proxy_gray.empty() &&
            !curr_frame->proxy_gray.empty() &&
            !prev_pts.empty()) {
          const auto t_klt_begin = Clock::now();

          std::vector<cv::Point2f> next_pts;
          std::vector<unsigned char> status;
          std::vector<float> err;

          const cv::Size win_size(21, 21);
          const int max_level = 3;
          const cv::TermCriteria criteria(
              cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);

          cv::calcOpticalFlowPyrLK(prev_frame->proxy_gray,
                                   curr_frame->proxy_gray,
                                   prev_pts,
                                   next_pts,
                                   status,
                                   err,
                                   win_size,
                                   max_level,
                                   criteria,
                                   0,
                                   1e-4);

          std::vector<cv::Point2f> good_prev;
          std::vector<cv::Point2f> good_curr;
          good_prev.reserve(next_pts.size());
          good_curr.reserve(next_pts.size());

          for (std::size_t i = 0; i < next_pts.size(); ++i) {
            if (!status[i]) {
              continue;
            }
            if (err[i] > static_cast<float>(config_.max_klt_error)) {
              continue;
            }
            const cv::Point2f& p0 = prev_pts[i];
            const cv::Point2f& p1 = next_pts[i];

            if (p1.x < 0 || p1.y < 0 ||
                p1.x >= curr_frame->proxy_gray.cols ||
                p1.y >= curr_frame->proxy_gray.rows) {
              continue;
            }

            good_prev.push_back(p0);
            good_curr.push_back(p1);
          }

          const auto t_klt_end = Clock::now();
          klt_ms = MsSince(t_klt_begin, t_klt_end);

          if (static_cast<int>(good_curr.size()) >= config_.min_pts_for_ransac) {
            const auto t_ransac_begin = Clock::now();

            Eigen::Vector2d delta;
            double noise = 0.0;
            std::vector<int> inliers;
            const bool ok = EstimateTranslationRansac(good_prev,
                                                      good_curr,
                                                      &delta,
                                                      &noise,
                                                      &inliers);

            const auto t_ransac_end = Clock::now();
            ransac_ms = MsSince(t_ransac_begin, t_ransac_end);

            if (ok) {
              track_res.delta_valid = true;
              track_res.delta = delta;
              track_res.delta_noise = noise;

              // 以 inlier 的 curr 点作为新的 tracked points + global center
              curr_pts.reserve(inliers.size());
              double sum_x = 0.0;
              double sum_y = 0.0;
              for (int idx : inliers) {
                const auto& pt = good_curr[static_cast<std::size_t>(idx)];
                curr_pts.push_back(pt);
                sum_x += static_cast<double>(pt.x);
                sum_y += static_cast<double>(pt.y);
              }
              const int cnt = static_cast<int>(curr_pts.size());
              if (cnt > 0) {
                track_res.global_center_valid = true;
                track_res.global_center =
                    Eigen::Vector2d(sum_x / cnt, sum_y / cnt);
              }

              LOG(INFO) << "[Tracker] Frame " << frame_idx
                        << " KLT+RANSAC ok, inliers=" << curr_pts.size()
                        << ", delta=(" << delta.x() << "," << delta.y()
                        << "), noise=" << noise;
            } else {
              LOG(INFO) << "[Tracker] Frame " << frame_idx
                        << " KLT+RANSAC failed. good_pts="
                        << good_curr.size();
              curr_pts.clear();
            }
          } else {
            LOG(INFO) << "[Tracker] Frame " << frame_idx
                      << " too few good KLT points: "
                      << good_curr.size() << " < "
                      << config_.min_pts_for_ransac;
            curr_pts.clear();
          }
        }

        // ---------------- 2) YOLO 结果：初始化 / 纠错重置 ----------------
        SegDetectorRes seg_res;
        const bool has_seg =
            seg_detect_worker_->QueryResult(frame_idx, &seg_res);
        
        if (has_seg) {
          track_res.seg_detect_res = seg_res;
        }

        if (has_seg && seg_res.has_select_object) {
          const auto& box = seg_res.select_object.box;
          const double bbox_cx =
              static_cast<double>(box.x) +
              0.5 * static_cast<double>(box.width);
          const double bbox_cy =
              static_cast<double>(box.y) +
              0.5 * static_cast<double>(box.height);

          // 2.1 如果没有任何跟踪点，直接用 YOLO good_features 初始化
          if (curr_pts.empty()) {
            curr_pts = seg_res.good_pts_to_track;

            track_res.global_center_valid = true;
            track_res.global_center = Eigen::Vector2d(bbox_cx, bbox_cy);

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
                track_res.global_center =
                    Eigen::Vector2d(bbox_cx, bbox_cy);

                // 本帧刚 re-init，delta 不可信。
                track_res.delta_valid = false;

                LOG(INFO) << "[Tracker] Frame " << frame_idx
                          << " re-init KLT from YOLO, pts=" << curr_pts.size()
                          << ", center=(" << bbox_cx << "," << bbox_cy << ")";
              }
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

        // ---------------- 4) 统计总耗时并打印 ----------------
        const auto t_frame_end = Clock::now();
        const double total_ms = MsSince(t_frame_start, t_frame_end);

        LOG(INFO) << "[Tracker] Frame " << frame_idx
                  << " timing_ms { wait_yolo=" << wait_yolo_ms
                  << ", klt=" << klt_ms
                  << ", ransac=" << ransac_ms
                  << ", total=" << total_ms << " }"
                  << ", delta_valid=" << track_res.delta_valid
                  << ", delta=(" << track_res.delta.x() << ","
                  << track_res.delta.y() << ")"
                  << ", pts_prev=" << prev_pts.size()
                  << ", has_seg=" << has_seg
                  << ", has_select_obj="
                  << (has_seg && seg_res.has_select_object);
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
