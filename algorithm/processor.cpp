#include "algorithm/processor.h"

#include <utility>
#include <glog/logging.h>

#include "common/file_utils.hpp"

namespace airsteady {

Processor::Processor(const Config& config) : config_(config) {
  std::string video_name;
  GetVideoName(config_.video_path, &video_name);

  const std::string video_work_folder = config_.system_params.work_folder + "/" + video_name;
  CreateFolder(video_work_folder);

  const std::string proxy_path = video_work_folder + "/proxy.mp4";
  video_preprocessor_ = std::make_shared<VideoPreprocessor>(
    config_.video_path,
    proxy_path,
    config_.system_params.max_porxy_resolution
  );

  tracker_ = std::make_shared<Tracker>(video_preprocessor_.get());
  // Add finished callback.
  tracker_->AddTrackFinishedCallback(std::bind(&Processor::OnTrackingFinished, this));

  StabilizerConfig stabilizer_cfg;
  stabilizer_cfg.debug_output_dir = video_work_folder;
  stabilizer_ = std::make_shared<Stabilizer>(stabilizer_cfg);

  // 预览器
  previewer_ = std::make_shared<Previewer>(proxy_path); 

  // Idle
  SetStatus(Status::kIdle);
}

Processor::Processor(std::string work_folder) : work_folder_(std::move(work_folder)) {}

Processor::~Processor() = default;

bool Processor::TryOpenVideo(std::string* err_info) {
  LOG(INFO) << "Try to open video: " << config_.video_path;
  bool ret = video_preprocessor_->TryOpenVideo(err_info);
  if (ret) {
    video_info_ = video_preprocessor_->GetVideoInfo();
  }
  LOG(INFO) << video_info_;
  return ret;
}

void Processor::StopAll() {}

void Processor::SetStatus(const Status& status) {
  status_.store(status, std::memory_order_release);
}

bool Processor::GetVideoInfo(VideoInfo* video_info) const {
  if (video_info == nullptr) {
    return false;
  }
  *video_info = video_info_;

  return true;
}

bool Processor::StartTracking(std::string* err_info) {
  bool ret = tracker_->StartTracking();

  if (ret) {
    SetStatus(Status::kTracking);
  }
  return ret;
}

void Processor::StopTracking() {}

void Processor::AddTrackingResultCallback(TrackingResultCallback cb) {
  tracker_->AddTrackingResultCallback(cb);
}

void Processor::AddTrackFinishedCallback(TrackFinishedCallback cb) {
  tracker_->AddTrackFinishedCallback(cb);
}

bool Processor::StartStabilize(std::string* err_info) {
  (void)err_info;
  stabilizer_->StartStabilize(tracker_->GetTrackingResults(), 
                              video_info_,
                              config_.stable_params.smooth_ratio);
  return true;
}

void Processor::AddStableFinishedCallback(StableFinishedCallback cb) {
  stable_finished_cbs_.push_back(std::move(cb));
}

bool Processor::UpdateParamAndRestable(const StableParams& stable_params, std::string* err_info) {
  (void)stable_params;
  (void)err_info;

  config_.stable_params = stable_params;

  return StartStabilize(err_info);
}

bool Processor::StartPreview() {
  return previewer_->StartPreview();
}

bool Processor::HoldPreview() {
  return previewer_->HoldPreview();
}

void Processor::SeekPreview(int frame_idx) {
  return previewer_->SeekPreview(frame_idx);
}

void Processor::AddPreviewCallback(PreviewCallback cb) {
  previewer_->AddPreviewCallback(cb);
}

bool Processor::StartExport(std::string* err) {
  (void)err;
  return false;
}

bool Processor::AddExportCallback(ExportProgressCallback cb) {
  export_progress_cbs_.push_back(std::move(cb));
  return true;
}

bool Processor::Save(const std::string& work_folder) {
  (void)work_folder;
  return false;
}

bool Processor::Load(std::string& work_folder) {
  (void)work_folder;
  return false;
}

void Processor::OnTrackingFinished() {
  SetStatus(Status::kStabilizing);

  LOG(WARNING) << "TrackEnd Start stablize!!!";
  std::string err;
  StartStabilize(&err);
}

void Processor::OnStabilizerFinished() {
  LOG(INFO) << "Stabilizer finished, insert result to previewer.";
  previewer_->SetTrackResults(tracker_->GetTrackingResults(),
              stabilizer_->GetStabilizedResults());
  
  // Show the first frame.
  previewer_->SeekAndPreviewOnce(0);

  SetStatus(Status::kStabilized);
}

}  // namespace airsteady
