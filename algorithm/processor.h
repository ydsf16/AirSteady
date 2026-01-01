#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace airsteady {

// Forward declarations (avoid heavy headers here).
struct FrameView;     // Your lightweight image view struct.
struct TrackRes;      // Per-frame tracking result.
struct StableParam;   // Per-frame stabilization parameters.

// One video corresponds to one Processor instance.
class Processor {
 public:
  // Configuration from UI / task queue.
  struct Config {
    // Folder to store all intermediate data for this video.
    std::string work_folder;

    // Input video path.
    std::string video_path;

    // Stabilization strength.
    double stable_ratio = 0.8;

    // Crop / transform parameters.
    bool enable_transform = false;
    double crop_ratio = 0.9;
    double offset_x = 0.0;
    double offset_y = 0.0;

    // Proxy-related parameters.
    // e.g. 1080 means the longer side of proxy frames is at most 1080.
    int max_proxy_resolution = 1080;

    // Export-related parameters.
    // If empty, Processor may use a default file name in work_folder.
    std::string export_path;

    // 0 means "auto" or "same as source".
    double export_bitrate = 0.0;

    // 0 means "same as source".
    int export_resolution = 0;
  };

  enum class Stage {
    kIdle = 0,
    kVideoOpened,
    kTracking,
    kTrackingFailed,
    kTrackingFinished,
    kStabilizing,
    kStabilizingFailed,
    kStabilizingFinished,
    kRecomputing,
    kRecomputingFinished,
    kExporting,
    kExportFailed,
    kFinished,
  };

  struct VideoInfo {
    bool valid = false;

    int width = 0;
    int height = 0;
    double bitrate = 0.0;

    double total_time_sec = 0.0;
    std::int64_t num_frames = 0;

    // TODO: camera intrinsics / extrinsics if available.
    // TODO: codec / pixel format info if needed.
  };

  using FrameIndex = std::int64_t;

  // Callback types.
  using TrackingResultCallback =
      std::function<void(const FrameView& proxy_frame,
                         const TrackRes& track_res,
                         double finished_ratio)>;

  using TrackFinishedCallback = std::function<void()>;
  using StableFinishedCallback = std::function<void()>;

  // For preview: left is "tracking view", right is "stabilized view".
  // Either track_res or stable_param can be nullptr if not available.
  using PreviewCallback =
      std::function<void(const FrameView& proxy_frame,
                         const TrackRes* track_res_for_left,
                         const StableParam* stable_param_for_right)>;

  using ExportProgressCallback = std::function<void(double progress_ratio)>;

 public:
  // Construct from configuration, used when user clicks "Open Video".
  explicit Processor(const Config& config);

  // Construct from an existing work folder, used for re-editing.
  explicit Processor(std::string work_folder);

  Processor(const Processor&) = delete;
  Processor& operator=(const Processor&) = delete;

  ~Processor();

  // Try to open the video and probe basic info.
  // Returns false and fills err_info if failed.
  bool TryOpenVideo(std::string* err_info);

  // Stop all internal work (tracking / preview / export) and flush
  // intermediate results to work_folder if needed.
  void Stop();

  // Read-only accessors.
  const Config& config() const { return config_; }

  Stage stage() const { return stage_.load(std::memory_order_acquire); }

  bool GetVideoInfo(VideoInfo* video_info) const;

  // ---------------- Tracking ----------------

  // Start tracking asynchronously.
  // Internally this will start a worker thread that:
  //   - reads frames (possibly through a proxy decoder)
  //   - runs YOLO detections sparsely
  //   - runs per-frame tracking
  //   - writes proxy video / track results to work_folder
  //
  // This function returns immediately.
  bool StartTracking(std::string* err_info);

  // Request to stop tracking worker; blocks until worker exits.
  void StopTracking();

  // Register callbacks before StartTracking().
  void AddTrackingResultCallback(TrackingResultCallback cb);
  void AddTrackFinishedCallback(TrackFinishedCallback cb);

  // ---------------- Stabilization ----------------

  // Compute stabilization trajectory based on tracked results.
  // This is a blocking call; you may run it in a separate thread.
  bool ComputeStableTraj(std::string* err_info);

  void AddStableFinishedCallback(StableFinishedCallback cb);

  // Update parameters and recompute stabilization.
  // e.g. user changes stable_ratio / crop_ratio / offsets in UI.
  bool UpdateParamAndRecompute(double stable_ratio,
                               bool enable_transform,
                               double crop_ratio,
                               double offset_x,
                               double offset_y,
                               std::string* err_info);

  // ---------------- Preview ----------------

  // Prepare preview, e.g. open proxy decoder, seek to first frame, etc.
  bool PreparePreview(std::string* err_info);

  // Start preview playback asynchronously.
  bool StartPreview(std::string* err_info);

  // Stop preview playback and release related resources.
  void StopPreview();

  // Seek to a specific timestamp (in seconds).
  // Internally maps to nearest frame index and triggers PreviewCallback.
  bool Seek(double time_sec);

  // Register preview callback; usually called once after Processor creation.
  void AddPreviewCallback(PreviewCallback cb);

  // ---------------- Export ----------------

  // Export final stabilized video.
  //
  // If export_path_override is non-empty, it overrides config_.export_path.
  // progress_cb is optional and may be empty.
  //
  // This is designed as a blocking call; you probably want to call it
  // from a background thread so UI remains responsive.
  bool Export(std::string* err_info,
              const std::string& export_path_override,
              ExportProgressCallback progress_cb);

  // ---------------- Persistence ----------------

  // Save intermediate results into work_folder for later re-edit.
  bool SaveTrackResult(std::string* err_info);
  bool SaveStableResult(std::string* err_info);
  bool SaveMetaInfo(std::string* err_info);

 private:
  // Internal helpers (to be implemented in .cc).
  void SetStage(Stage new_stage);

  // Thread-safe dispatch for callbacks.
  void NotifyTrackingResult(const FrameView& frame,
                            const TrackRes& track_res,
                            double finished_ratio);
  void NotifyTrackFinished();
  void NotifyStableFinished();
  void NotifyPreview(const FrameView& frame,
                     const TrackRes* track_res,
                     const StableParam* stable_param);

 private:
  Config config_;

  // In case you construct from work_folder-only ctor.
  std::string work_folder_;

  // Stage is read/written from multiple threads.
  std::atomic<Stage> stage_{Stage::kIdle};

  VideoInfo video_info_;

  // Per-frame results indexed by frame index.
  std::unordered_map<FrameIndex, std::shared_ptr<TrackRes>> track_results_;
  std::unordered_map<FrameIndex, std::shared_ptr<StableParam>> stable_params_;

  // Callback registrations.
  // If you allow dynamic add/remove during processing,
  // protect these with a mutex in implementation.
  std::vector<TrackingResultCallback> tracking_result_cbs_;
  std::vector<TrackFinishedCallback> track_finished_cbs_;
  std::vector<StableFinishedCallback> stable_finished_cbs_;
  std::vector<PreviewCallback> preview_cbs_;

  // TODO: add worker threads / queues / decoder handles here.
  // e.g.
  // std::thread tracking_thread_;
  // std::thread preview_thread_;
  // std::thread export_thread_;
  //
  // std::atomic<bool> tracking_stop_requested_{false};
  // std::atomic<bool> preview_stop_requested_{false};
  // std::atomic<bool> export_stop_requested_{false};
};

}  // namespace airsteady
