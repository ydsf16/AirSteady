#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "common/types.h"

namespace airsteady {

class Previewer {
 public:
  explicit Previewer(const std::string& proxy_bgr_path);
  ~Previewer();

  Previewer(const Previewer&) = delete;
  Previewer& operator=(const Previewer&) = delete;

  // Initialize video decoding and start worker thread.
  // Must be called once before StartPreview / Seek / SeekAndPreviewOnce.
  // Returns false on failure, and optionally fills err_msg.
  bool Init(std::string* err_msg = nullptr);

  // Replace all tracking / stable results.
  void SetTrackResults(const std::vector<FrameTrackingResult>& track_results,
                       const std::vector<FrameStableResult>& stable_results);

  // Start preview playback asynchronously.
  bool StartPreview();

  // Pause preview. Returns true if it was playing before.
  bool HoldPreview();

  // Seek to an absolute frame index in proxy video.
  // Does not implicitly start playback; only changes the current frame.
  void SeekPreview(int frame_idx);

  // Seek to a frame and preview that frame once (synchronously).
  // Does NOT change playing/paused state; only shows a single static frame.
  // Blocks until the frame has been decoded and callbacks have run,
  // or until Previewer is shutting down.
  void SeekAndPreviewOnce(int frame_idx);

  // Register preview callback; usually called once after Processor creation.
  using PreviewCallback = std::function<void(const FramePreview& frame_preview)>;
  void AddPreviewCallback(PreviewCallback cb);

 private:
  void Run();  // Worker thread main loop.

  // Make a FramePreview for given frame & index.
  FramePreview MakeFramePreviewLocked(const cv::Mat& frame, int frame_idx)
      const;

 private:
  // Video.
  std::string proxy_bgr_path_;
  cv::VideoCapture cap_;
  double fps_ = 0.0;
  int total_frames_ = 0;

  // Per-frame results.
  std::unordered_map<int, FrameTrackingResult> frame_idx_track_results_;
  std::unordered_map<int, FrameStableResult> frame_idx_stable_results_;

  // Threading.
  mutable std::mutex mutex_;
  std::condition_variable cond_var_;
  std::thread worker_thread_;
  bool running_ = false;   // worker thread running flag
  bool playing_ = false;   // playback state

  bool seek_pending_ = false;
  int seek_frame_idx_ = 0;

  int current_frame_idx_ = 0;

  std::vector<PreviewCallback> callbacks_;

  // Single-preview (seek & show once) request, executed on worker thread.
  bool single_preview_pending_ = false;
  bool single_preview_ready_ = false;
  int single_preview_frame_idx_ = 0;
  FramePreview single_preview_;  // Reserved for future use.
  std::condition_variable single_preview_cv_;
};

}  // namespace airsteady
