#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common/types.h"

namespace airsteady {

class VideoExportor {
 public:
  VideoExportor();
  ~VideoExportor();

  // 内部开线程做导出。
  void StartExport(const std::string& src_video_path,
                   const VideoInfo& video_info,
                   const ExportParams& export_param,
                   const std::vector<FrameStableResult>& stable_results);

  using ExportProgressCallback = std::function<void(int frame_idx)>;
  void AddExportCallback(ExportProgressCallback cb);
  void AddExportDoneCallback(std::function<void()> cb);

  // 可选：外部请求停止（best-effort）。
  void Stop();

 private:
  void WorkerMain(std::string src_video_path,
                  VideoInfo video_info,
                  ExportParams export_param,
                  std::vector<FrameStableResult> stable_results);

 private:
  std::thread worker_;
  std::atomic<bool> stop_{false};
  std::atomic<bool> running_{false};

  std::mutex cb_mtx_;
  std::vector<ExportProgressCallback> progress_cbs_;
  std::vector<std::function<void()>> done_cbs_;
};

}