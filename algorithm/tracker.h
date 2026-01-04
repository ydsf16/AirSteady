#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "common/types.h"
#include "algorithm/video_preprocessor.h"
#include "yolo/yolo_seg_detector.h"
#include "algorithm/seg_detector_worker.h"

namespace airsteady {

class Tracker {
 public:
  explicit Tracker(VideoPreprocessor* video_preprocessor);
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

 private:
  VideoPreprocessor* video_preprocessor_ = nullptr;

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
