#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "common/types.h"
#include "algorithm/video_preprocessor.h"
#include "yolo/yolo_seg_detector.h"
#include "algorithm/seg_detector_worker.h"

namespace airsteady {

// Frame-to-frame tracking based on:
// 1) Sparse YOLO seg-based GFTT initialization (SegDetectorWorker)
// 2) KLT optical flow (PyrLK) for per-frame tracking
// 3) RANSAC-based robust translation estimation
class Tracker {
 public:
  // Runtime tuning parameters for KLT + RANSAC + YOLO re-init.
  struct Config {
    // How many frames to delay before consuming, so YOLO has time to produce results.
    int delay_n_frames = 30;

    // Minimum number of point correspondences required to run RANSAC.
    int min_pts_for_ransac = 2;

    // Inlier threshold (pixels) for translation RANSAC.
    double ransac_inlier_thresh = 1.0;

    // Minimum inlier ratio to accept the translation model.
    double ransac_min_inlier_ratio = 0.0;

    // If current tracked points < this, we consider tracking poor
    // and prefer YOLO re-init (when available).
    int reinit_pts_threshold = 400;

    // Max per-point KLT error to keep a correspondence.
    double max_klt_error = 10000.0;

    // For seg detector.
    int yolo_detect_every_n_frames = 15;
    int max_num_good_features = 1000;
    std::string select_obj_name = "airplane";

    int debug_draw_every_n = 0;
    std::string debug_draw_dir = "";
    bool debug_draw_show = true;
  };

  explicit Tracker(VideoPreprocessor* video_preprocessor,
                   const Config& config = Config());
  ~Tracker();

  bool StartTracking();
  void StopTracking();

  using TrackingResultCallback =
      std::function<void(const FrameTrackingResultPreview& res)>;
  using TrackFinishedCallback = std::function<void()>;

  void AddTrackingResultCallback(TrackingResultCallback cb);
  void AddTrackFinishedCallback(TrackFinishedCallback cb);

  std::vector<FrameTrackingResult> GetTrackingResults() const;

 private:
  void Run();

  // Robust translation estimation using simple RANSAC on point correspondences.
  // prev_pts[i] <-> curr_pts[i].
  bool EstimateTranslationRansac(const std::vector<cv::Point2f>& prev_pts,
                                 const std::vector<cv::Point2f>& curr_pts,
                                 Eigen::Vector2d* delta,
                                 double* noise,
                                 std::vector<int>* inlier_indices) const;

 private:
  VideoPreprocessor* video_preprocessor_ = nullptr;
  Config config_;

  // Protects track_results_.
  mutable std::mutex mutex_;
  std::vector<FrameTrackingResult> track_results_;

  std::shared_ptr<std::thread> thread_;
  std::atomic<bool> stop_{false};

  std::vector<TrackingResultCallback> tracking_result_cbs_;
  std::vector<TrackFinishedCallback> track_finished_cbs_;

  std::shared_ptr<yolo::YoloSegDetector> yolo_seg_detector_;
  std::shared_ptr<SegDetectorWorker> seg_detect_worker_;
};

}  // namespace airsteady
