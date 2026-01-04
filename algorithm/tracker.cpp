#include "algorithm/tracker.h"

#include <glog/logging.h>

#include "common/file_utils.hpp"

namespace airsteady {

Tracker::Tracker(VideoPreprocessor* video_preprocessor)
    : video_preprocessor_(video_preprocessor) {
  CHECK(video_preprocessor != nullptr);
    
  yolo::YoloConfig yolo_cfg;
  std::string exe_folder;
  GetExeFolder(&exe_folder);
  yolo_cfg.onnx_path = exe_folder  + "/model/model.onnx";

  LOG(INFO) << "onnx_path: " << yolo_cfg.onnx_path; 

  yolo_cfg.preferred_provider = airsteady::yolo::Provider::kDirectML;
  yolo_cfg.enable_auto_fallback = true;
  yolo_cfg.verbose_log = true;
  yolo_cfg.enable_profiling = true;

  yolo_seg_detector_ = std::make_shared<yolo::YoloSegDetector>(yolo_cfg);

  yolo_seg_detector_->SetEventCallback([](const airsteady::yolo::DetectorEvent& ev) {
    // 你可以接到 glog / fmt / Qt log panel
    // 这里只用 printf 举例
    // severity/code 你也可以映射成不同颜色
    LOG(INFO) << "yolo: " << static_cast<int>(ev.severity) << ", " << static_cast<int>(ev.code) << ", " <<
          ev.message.c_str() << ", " << 
          ev.from_provider.c_str()<< ", " <<  ev.to_provider.c_str()<< ", " << 
          ev.details.c_str();
  });

  SegDetectorWorker::Config seg_detect_worker_cfg;
  seg_detect_worker_ = 
    std::make_shared<SegDetectorWorker>(seg_detect_worker_cfg, yolo_seg_detector_);

  // std::string err;
  // if (!yolo_seg_detector_->Init(&err)) {
  //   LOG(FATAL) << "yolo failed!!!" << err.c_str(); 
  //   return;
  // }
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

    // Feed frame to seg detector worker.
    seg_detect_worker_->FeedFrame(frame);

    FrameTrackingResultPreview preview;
    preview.proxy_bgr = frame->proxy_bgr;

    for (auto func : tracking_result_cbs_) {
      func(preview);
    }
  }
}

}  // namespace airsteady
