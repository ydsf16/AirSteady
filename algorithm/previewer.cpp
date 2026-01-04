#include "algorithm/previewer.h"

#include <algorithm>
#include <chrono>

namespace airsteady {

namespace {

int ClampFrameIndex(int idx, int total_frames) {
  if (total_frames <= 0) {
    return std::max(idx, 0);
  }
  if (idx < 0) {
    return 0;
  }
  if (idx >= total_frames) {
    return total_frames - 1;
  }
  return idx;
}

}  // namespace

Previewer::Previewer(const std::string& proxy_bgr_path)
    : proxy_bgr_path_(proxy_bgr_path) {
  // Do NOT open video or start thread here.
  // Call Init() explicitly after construction.
}

Previewer::~Previewer() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    running_ = false;
    playing_ = false;
    seek_pending_ = false;
    single_preview_pending_ = false;
  }
  cond_var_.notify_all();
  single_preview_cv_.notify_all();

  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }
}

bool Previewer::Init(std::string* err_msg) {
  // Quick check without lock to avoid re-init.
  if (running_) {
    return true;
  }

  // Open video before starting worker thread (no other thread touches cap_ yet).
  cap_.open(proxy_bgr_path_);
  if (!cap_.isOpened()) {
    if (err_msg != nullptr) {
      *err_msg = "Failed to open proxy video: " + proxy_bgr_path_;
    }
    return false;
  }

  fps_ = cap_.get(cv::CAP_PROP_FPS);
  if (fps_ <= 0.0) {
    // Fallback: treat as 30fps if metadata is broken.
    fps_ = 30.0;
  }
  total_frames_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_COUNT) + 0.5);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (running_) {
      // Another thread might have called Init successfully in the meantime.
      return true;
    }
    running_ = true;
    playing_ = false;
    seek_pending_ = false;
    single_preview_pending_ = false;
    single_preview_ready_ = false;
    current_frame_idx_ = 0;
  }

  worker_thread_ = std::thread(&Previewer::Run, this);
  return true;
}

void Previewer::SetTrackResults(
    const std::vector<FrameTrackingResult>& track_results,
    const std::vector<FrameStableResult>& stable_results) {
  std::lock_guard<std::mutex> lock(mutex_);
  frame_idx_track_results_.clear();
  for (const auto& r : track_results) {
    frame_idx_track_results_[r.frame_idx] = r;
  }
  frame_idx_stable_results_.clear();
  for (const auto& r : stable_results) {
    frame_idx_stable_results_[r.frame_idx] = r;
  }
}

bool Previewer::StartPreview() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!running_ || !cap_.isOpened()) {
    return false;
  }

  // If already at the end, rewind to the beginning.
  if (total_frames_ > 0 && current_frame_idx_ >= total_frames_) {
    seek_pending_ = true;
    seek_frame_idx_ = 0;
  }

  playing_ = true;
  cond_var_.notify_all();
  return true;
}

bool Previewer::HoldPreview() {
  std::lock_guard<std::mutex> lock(mutex_);
  bool was_playing = playing_;
  playing_ = false;
  return was_playing;
}

void Previewer::SeekPreview(int frame_idx) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!running_ || !cap_.isOpened()) {
    current_frame_idx_ = ClampFrameIndex(frame_idx, total_frames_);
    return;
  }

  const int clamped = ClampFrameIndex(frame_idx, total_frames_);
  seek_pending_ = true;
  seek_frame_idx_ = clamped;
  cond_var_.notify_all();
}

void Previewer::SeekAndPreviewOnce(int frame_idx) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!running_ || !cap_.isOpened()) {
    return;
  }

  const int clamped = ClampFrameIndex(frame_idx, total_frames_);
  single_preview_pending_ = true;
  single_preview_ready_ = false;
  single_preview_frame_idx_ = clamped;

  cond_var_.notify_all();

  // Block until worker thread finishes decoding + callbacks.
  single_preview_cv_.wait(lock, [this]() {
    return single_preview_ready_ || !running_;
  });
}

void Previewer::AddPreviewCallback(PreviewCallback cb) {
  std::lock_guard<std::mutex> lock(mutex_);
  callbacks_.push_back(std::move(cb));
}

FramePreview Previewer::MakeFramePreviewLocked(const cv::Mat& frame,
                                               int frame_idx) const {
  FramePreview preview;
  preview.frame_idx = frame_idx;

  if (fps_ > 0.0) {
    const double t_sec = static_cast<double>(frame_idx) / fps_;
    preview.time_ns = static_cast<std::int64_t>(t_sec * 1e9);
  } else {
    preview.time_ns = 0;
  }

  // Clone to ensure lifetime beyond this function.
  preview.proxy_bgr = frame.clone();

  const auto track_it = frame_idx_track_results_.find(frame_idx);
  if (track_it != frame_idx_track_results_.end()) {
    preview.track_res = track_it->second;
    if (preview.time_ns == 0) {
      preview.time_ns = track_it->second.time_ns;
    }
  }

  const auto stable_it = frame_idx_stable_results_.find(frame_idx);
  if (stable_it != frame_idx_stable_results_.end()) {
    preview.stable_res = stable_it->second;
    if (preview.time_ns == 0) {
      preview.time_ns = stable_it->second.time_ns;
    }
  }

  return preview;
}

void Previewer::Run() {
  cv::Mat frame;

  using Clock = std::chrono::steady_clock;
  const auto frame_duration =
      std::chrono::duration<double>(1.0 / std::max(fps_, 1e-6));

  while (true) {
    int local_frame_idx = -1;
    bool local_playing = false;
    bool local_seek_pending = false;
    int local_seek_frame_idx = 0;

    bool do_single_preview = false;
    int single_idx = 0;

    {
      std::unique_lock<std::mutex> lock(mutex_);
      cond_var_.wait(lock, [this]() {
        return !running_ || playing_ || seek_pending_ ||
               single_preview_pending_;
      });

      if (!running_) {
        break;
      }

      if (single_preview_pending_) {
        do_single_preview = true;
        single_idx = single_preview_frame_idx_;
      } else {
        local_seek_pending = seek_pending_;
        local_seek_frame_idx = seek_frame_idx_;
        local_playing = playing_;

        if (local_seek_pending && cap_.isOpened()) {
          cap_.set(cv::CAP_PROP_POS_FRAMES, local_seek_frame_idx);
          current_frame_idx_ = local_seek_frame_idx;
          seek_pending_ = false;
        }

        if (!local_playing) {
          continue;  // paused but maybe just handled seek.
        }

        local_frame_idx = current_frame_idx_;
      }
    }

    if (do_single_preview) {
      // Single preview path: seek to a specific frame, decode, call callbacks.
      cap_.set(cv::CAP_PROP_POS_FRAMES, single_idx);

      if (!cap_.read(frame) || frame.empty()) {
        std::lock_guard<std::mutex> lock(mutex_);
        single_preview_ = FramePreview{};
        single_preview_.frame_idx = single_idx;
        single_preview_ready_ = true;
        single_preview_pending_ = false;
        single_preview_cv_.notify_all();
        continue;
      }

      FramePreview preview;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        preview = MakeFramePreviewLocked(frame, single_idx);
        single_preview_ = preview;
        current_frame_idx_ = single_idx;
      }

      std::vector<PreviewCallback> callbacks_copy;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        callbacks_copy = callbacks_;
      }
      for (const auto& cb : callbacks_copy) {
        if (cb) {
          cb(preview);
        }
      }

      {
        std::lock_guard<std::mutex> lock(mutex_);
        single_preview_ready_ = true;
        single_preview_pending_ = false;
      }
      single_preview_cv_.notify_all();

      // Reset decode position to this frame so next StartPreview continues here.
      cap_.set(cv::CAP_PROP_POS_FRAMES, single_idx);

      continue;
    }

    // Normal playback path.
    const auto start_time = Clock::now();

    if (!cap_.read(frame) || frame.empty()) {
      std::lock_guard<std::mutex> lock(mutex_);
      playing_ = false;
      continue;
    }

    FramePreview preview;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      preview = MakeFramePreviewLocked(frame, local_frame_idx);
    }

    std::vector<PreviewCallback> callbacks_copy;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      callbacks_copy = callbacks_;
    }

    for (const auto& cb : callbacks_copy) {
      if (cb) {
        cb(preview);
      }
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      ++current_frame_idx_;
    }

    // Control frame pacing: subtract decode + callback time.
    const auto elapsed = Clock::now() - start_time;
    if (elapsed < frame_duration) {
      std::this_thread::sleep_for(
          std::chrono::duration_cast<std::chrono::milliseconds>(
              frame_duration - elapsed));
    }
  }
}

}  // namespace airsteady
