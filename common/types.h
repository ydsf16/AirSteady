#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <ostream>

#include <Eigen/Core>
#include <opencv2/core/mat.hpp>

#include "yolo/yolo_types.h"

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


struct FrameStableResult {
  int frame_idx = 0;
  std::int64_t time_ns = 0;

  double delta_u = 0.0;
  double delta_v = 0.0;
  double delta_yaw = 0.0;
};

struct PreFrame {
  int64_t time_ns = 0;
  int frame_idx = 0;
  cv::Mat proxy_bgr;
  cv::Mat proxy_gray;
};


// Per-frame timing stats for debugging and profiling.
struct SegDetectorTiming {
  double yolo_ms = 0.0;           // YOLO inference time.
  double select_object_ms = 0.0;  // Object selection + GFTT pipeline time.
  double gftt_ms = 0.0;           // goodFeaturesToTrack time (subset of select).
  double total_ms = 0.0;          // End-to-end time inside RunDetection.
};

struct SegDetectorRes {
  int frame_idx = -1;
  std::int64_t time_ns = 0;

  // All detections in this frame.
  std::vector<yolo::Det> yolo_objects;

  // Selected tracking target.
  bool has_select_object = false;
  yolo::Det select_object;
  // Good features (float) inside target (after mask filtering).
  std::vector<cv::Point2f> good_pts_to_track;

  // Timing stats.
  SegDetectorTiming timing;
};

struct FrameTrackingResult {
  int frame_idx = 0;
  int time_ns = 0;

  // Delta frame result.
  bool delta_valid = false;
  Eigen::Vector2d delta = Eigen::Vector2d::Zero();
  double delta_noise = 0.0;

  // Global center.
  bool global_center_valid = false;
  Eigen::Vector2d global_center = Eigen::Vector2d::Zero();
};

struct FrameTrackingResultPreview {
  SegDetectorRes seg_detect_res;
  cv::Mat proxy_bgr;
};

struct FramePreview {
  int frame_idx = 0;
  std::int64_t time_ns = 0;

  cv::Mat proxy_bgr;

  FrameTrackingResult track_res;
  FrameStableResult stable_res;
};

}  // namespace airsteady
