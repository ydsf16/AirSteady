#include "algorithm/tracker.h"

#include <glog/logging.h>

namespace airsteady {

Tracker::Tracker(VideoPreprocessor* video_preprocessor)
    : video_preprocessor_(video_preprocessor) {
  CHECK(video_preprocessor != nullptr);
}

Tracker::~Tracker() {
  StopTracking();
}

bool Tracker::StartTracking() {
  // 标记运行
  stop_.store(false, std::memory_order_relaxed);

  // 启动线程：成员函数指针 + this
  thread_ = std::make_shared<std::thread>(&Tracker::Run, this);
}

void Tracker::StopTracking() {
  stop_.store(true, std::memory_order_relaxed);

  if (thread_ && thread_->joinable()) {
    thread_->join();
  }
}

void Tracker::AddTrackingResultCallback(TrackingResultCallback cb) {
  tracking_result_cbs_.push_back(cb);
}

void Tracker::AddTrackFinishedCallback(TrackFinishedCallback cb) {
  track_finished_cbs_.push_back(cb);
}

std::vector<FrameTrackingResult> Tracker::GetTrackingResults() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return track_results_;
}

void Tracker::Run() {
  while (!stop_.load(std::memory_order_relaxed)) {
    auto frame = video_preprocessor_->NextFrame();
    if (frame == nullptr) {
      stop_.store(true, std::memory_order_relaxed);
      LOG(INFO) << "Track Finished!!!";
      break;
    }

    // TODO: 在这里做真正的跟踪，并往 track_results_ 里 push 结果
    LOG(INFO) << "GetFrame: " << frame->time_ns << ", " << frame->frame_idx;

    FrameTrackingResultPreview preview;
    preview.proxy_bgr = frame->proxy_bgr;

    for (auto func : tracking_result_cbs_) {
      func(preview);
    }
  }
}

}  // namespace airsteady
