#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <ostream>

#include <Eigen/Core>
#include <opencv2/core/mat.hpp>

namespace airsteady {

struct StableParams {
  std::string track_type = "air_plane";

  double smooth_ratio = 1.0;  // 0.0 - 1.0

  bool enable_crop = true;
  double crop_keep_ratio = 1.0;
  double offset_u = 0.0;
  double offset_v = 0.0;
};

struct ExportParams {
  std::string export_path;
  double export_bitrate = 0.0;
  int export_resolution = 0;
};

struct SystemParams {
  std::string work_folder;
  int max_porxy_resolution = 1080;  // keep original name to avoid breaking callers
};

struct ProxyInfo {
  int width = 0;
  int height = 0;
  double scale = 1.0;

  double total_time_sec = 0.0;
  int num_frames = 0;
};

struct VideoInfo {
  int width = 0;
  int height = 0;
  double bitrate = 0.0;

  double total_time_sec = 0.0;
  int num_frames = 0;

  std::string codec;
};

inline std::ostream& operator<<(std::ostream& os, const VideoInfo& info) {
  os << "VideoInfo { "
     << "width: " << info.width << ", "
     << "height: " << info.height << ", "
     << "bitrate: " << info.bitrate << ", "
     << "total_time_sec: " << info.total_time_sec << ", "
     << "num_frames: " << info.num_frames << ", "
     << "codec: \"" << info.codec << "\""
     << " }";
  return os;
}

struct TrackResult {
  double var = 0.0;
};

struct BBox {
  double center_u = 0.0;
  double center_v = 0.0;
  double width = 0.0;
  double height = 0.0;
};

struct FrameTrackingResult {
  int frame_idx = 0;
  std::int64_t time_ns = 0;

  // Delta frame result.
  bool delta_valid = false;
  Eigen::Vector2d delta = Eigen::Vector2d::Zero();
  double delta_noise = 0.0;

  // Global center.
  bool global_center_valid = false;
  Eigen::Vector2d global_center = Eigen::Vector2d::Zero();
};

struct FrameTrackingResultPreview {
  FrameTrackingResult track_res;

  cv::Mat proxy_bgr;
  std::vector<BBox> bboxes;

  double progress_ratio = 0.0;
};

struct FrameStableResult {
  int frame_idx = 0;
  std::int64_t time_ns = 0;

  double delta_u = 0.0;
  double delta_v = 0.0;
  double delta_yaw = 0.0;
};

struct FramePreview {
  int frame_idx = 0;
  std::int64_t time_ns = 0;

  cv::Mat proxy_bgr;

  FrameTrackingResult track_res;
  FrameStableResult stable_res;
};

struct PreFrame {
  int64_t time_ns = 0;
  int frame_idx = 0;
  cv::Mat proxy_bgr;
  cv::Mat proxy_gray;
};

}  // namespace airsteady
