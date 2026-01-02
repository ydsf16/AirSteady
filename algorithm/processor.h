#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common/types.h"
#include "algorithm/video_preprocessor.h"

namespace airsteady {

// One video corresponds to one Processor instance.
class Processor {
 public:
  struct Config {
    std::string video_path;

    SystemParams system_params;
    StableParams stable_params;
    ExportParams export_params;
  };

  enum Status : std::int8_t {
    kIdle = 0,
    kTracking,
    kStabilizing,
    kStabilized,
    kExporting,
    kFinished,
  };

  using TrackingResultCallback = std::function<void(const FrameTrackingResultPreview& res)>;
  using TrackFinishedCallback = std::function<void()>;
  using StableFinishedCallback = std::function<void()>;
  using PreviewCallback = std::function<void(const FramePreview& frame_preview)>;
  using ExportProgressCallback = std::function<void(double progress_ratio)>;

 public:
  // Construct from configuration, used when user clicks "Open Video".
  explicit Processor(const Config& config);
  // Construct from an existing work folder, used for re-editing.
  explicit Processor(std::string work_folder);
  ~Processor();

  // Try to open the video and probe basic info.
  // Returns false and fills err_info if failed.
  bool TryOpenVideo(std::string* err_info);

  // Stop all internal work (tracking / preview / export) and flush
  // intermediate results to work_folder if needed.
  void StopAll();

  // Read-only accessors.
  const Config& config() const { return config_; }

  void SetStatus(const Status& status);
  Status status() const { return status_.load(std::memory_order_acquire); }

  bool GetVideoInfo(VideoInfo* video_info) const;

  // ---------------- Tracking ----------------
  // Start tracking asynchronously.
  bool StartTracking(std::string* err_info);

  // Request to stop tracking worker; blocks until worker exits.
  void StopTracking();

  // Register callbacks before StartTracking().
  void AddTrackingResultCallback(TrackingResultCallback cb);
  void AddTrackFinishedCallback(TrackFinishedCallback cb);

  // ---------------- Stabilization ----------------
  bool StartStabilize(std::string* err_info);
  void AddStableFinishedCallback(StableFinishedCallback cb);
  bool UpdateParamAndRestable(const StableParams& stable_params, std::string* err_info);

  // ---------------- Preview ----------------
  // Prepare preview, e.g. open proxy decoder, seek to first frame, etc.
  bool PreparePreview(std::string* err_info);
  // Start preview playback asynchronously.
  bool StartPreview(std::string* err_info);
  // Stop preview playback and release related resources.
  void StopPreview();
  // Seek to a specific timestamp (in seconds).
  void SeekPreview(double time_sec);
  // Register preview callback; usually called once after Processor creation.
  void AddPreviewCallback(PreviewCallback cb);

  // ---------------- Export ----------------
  bool StartExport(std::string* err);
  bool AddExportCallback(ExportProgressCallback cb);

  // ---------------- Persistence ----------------
  bool Save(const std::string& work_folder);
  bool Load(std::string& work_folder);

 private:
  Config config_;

  // In case you construct from work_folder-only ctor.
  std::string work_folder_;

  // Stage is read/written from multiple threads.
  std::atomic<Status> status_{kIdle};

  VideoInfo video_info_;
  ProxyInfo proxy_info_;

  // Per-frame results indexed by frame index.
  // NOTE: key type is kept as int to match your original definition.
  std::map<int, std::shared_ptr<FrameTrackingResult>> time_ns_track_results_;
  std::map<int, std::shared_ptr<FrameStableResult>> time_ns_stable_results_;

  // Callback registrations.
  std::vector<TrackingResultCallback> tracking_result_cbs_;
  std::vector<TrackFinishedCallback> track_finished_cbs_;
  std::vector<StableFinishedCallback> stable_finished_cbs_;
  std::vector<PreviewCallback> preview_cbs_;
  std::vector<ExportProgressCallback> export_progress_cbs_;

  // Subprocessor.s
  std::shared_ptr<VideoPreprocessor> video_preprocessor_;
};

}  // namespace airsteady
