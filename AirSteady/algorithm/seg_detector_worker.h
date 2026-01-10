#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>

#include "common/types.h"
#include "yolo/yolo_types.h"
#include "yolo/yolo_seg_detector.h"

namespace airsteady {

class SegDetectorWorker {
 public:
  struct Config {
    int detect_every_n_frames = -1;

    std::size_t max_num_good_features = 100;
    std::string select_obj_name = "airplane";

    // Debug draw
    bool enable_debug = false;
    std::string debug_window_name = "SegDetectorDebug";
  };

  SegDetectorWorker(const Config& config,
                    std::shared_ptr<yolo::YoloSegDetector> yolo_seg_detector);
  ~SegDetectorWorker();

  SegDetectorWorker(const SegDetectorWorker&) = delete;
  SegDetectorWorker& operator=(const SegDetectorWorker&) = delete;

  // Feed high-frequency frames from outside (producer side).
  void FeedFrame(const std::shared_ptr<PreFrame>& pre_frame);

  // Max frame index that has been consumed by worker thread (including skipped).
  int GetProcessedFrameIdx() const;

  // Query YOLO + GFTT result for a given frame index.
  // Returns true if result exists.
  bool QueryResult(int frame_idx, SegDetectorRes* res) const;

  // Stop worker thread immediately (best-effort).
  // After Stop() returns, no more frames will be processed.
  void Stop();

 private:
  // Internal worker thread main loop.
  void WorkerLoop();

  // Whether we should run YOLO on this frame index.
  bool ShouldRunDetect(int frame_idx);

  // Run YOLO and fill SegDetectorRes (including selecting object + GFTT).
  void RunDetection(const std::shared_ptr<PreFrame>& pre_frame,
                    SegDetectorRes* out_res);

  // Select target object & extract good features to track.
  void SelectObjectAndExtractGftt(const cv::Mat& proxy_bgr,
                                  const cv::Mat& proxy_gray,
                                  const yolo::YoloResult& det_res,
                                  SegDetectorRes* out_res);

  // Trim result cache to avoid unbounded growth.
  void TrimResultCacheIfNeeded();

  void RenderDebugImage(const cv::Mat& proxy_bgr,
                        const SegDetectorRes& res);

 private:
  static constexpr std::size_t kMaxInputBufferSize = 256;
  static constexpr std::size_t kMaxResultCacheSize = 1024;

  Config config_;
  std::shared_ptr<yolo::YoloSegDetector> yolo_seg_detector_;

  // Input buffer (frames) and control.
  mutable std::mutex in_mutex_;
  std::condition_variable in_cv_;
  std::deque<std::shared_ptr<PreFrame>> pre_frame_buffer_;

  // Results cache: frame_idx → result.
  mutable std::mutex out_mutex_;
  std::unordered_map<int, std::shared_ptr<SegDetectorRes>> frame_idx_result_map_;
  std::deque<int> result_frame_order_;  // keep insertion order for trimming

  // Worker thread.
  std::thread worker_thread_;
  std::atomic<bool> stop_requested_{false};

  // Max processed frame index (includes non-detected frames).
  std::atomic<int> processed_frame_idx_{-1};

  // Last frame index on which YOLO was actually run (owned by worker thread).
  int last_detect_frame_idx_ = -1;
};

}  // namespace airsteady
