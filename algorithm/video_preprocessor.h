#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <opencv2/opencv.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/hwcontext.h>
}

#include "common/types.h"  // 假定这里定义了 VideoInfo

namespace airsteady {

struct PreFrame {
  int64_t time_ns = 0;
  int frame_idx = 0;
  cv::Mat proxy_bgr;
  cv::Mat proxy_gray;
};

// 负责：
// 1. 打开源视频，优先用硬件解码，失败则回退 CPU 解码
// 2. 每帧 resize 到 proxy 尺寸（CPU sws_scale，保证宽高为偶数）
// 3. 写入 proxy 视频（优先硬件编码，失败则 libx264/h264/mpeg4）
// 4. 在后台线程预取到一个有界队列中，NextFrame() 阻塞消费
class VideoPreprocessor {
 public:
  VideoPreprocessor(const std::string& src_video_path,
                    const std::string& proxy_save_path,
                    int max_proxy_resolution,
                    size_t num_prefetch = 10);

  ~VideoPreprocessor();

  // Return false on failure, err filled if non-null.
  bool TryOpenVideo(std::string* err);
  VideoInfo GetVideoInfo() const;

  // Blocking; returns nullptr on EOF or stop.
  std::shared_ptr<PreFrame> NextFrame();

 private:
  void PrefetchThread();
  bool InitFFmpeg(std::string* err);
  bool DecodeOneFrame(AVFrame** out_frame);

  // HW decode helpers.
  static enum AVPixelFormat GetHwFormat(struct AVCodecContext* ctx,
                                        const enum AVPixelFormat* pix_fmts);

 private:
  // Params.
  std::string src_path_;
  std::string proxy_path_;
  int max_proxy_res_;
  size_t max_prefetch_;

  // Threading.
  std::atomic<bool> stop_{false};
  std::atomic<bool> eof_{false};
  std::thread worker_;

  std::mutex mtx_;
  std::condition_variable cv_not_full_;
  std::condition_variable cv_not_empty_;
  std::deque<std::shared_ptr<PreFrame>> queue_;

  // FFmpeg decode.
  AVFormatContext* fmt_ = nullptr;
  AVCodecContext* dec_ctx_ = nullptr;
  AVStream* video_st_ = nullptr;
  int video_stream_idx_ = -1;

  AVPacket* pkt_ = nullptr;
  AVFrame* frame_ = nullptr;      // May be HW frame.
  AVFrame* sw_frame_ = nullptr;   // SW frame (for HW decode transfer).
  SwsContext* sws_ = nullptr;     // CPU scaler (decode → proxy BGR).

  // HW decode state.
  AVBufferRef* hw_device_ctx_ = nullptr;
  AVPixelFormat hw_pix_fmt_ = AV_PIX_FMT_NONE;
  bool using_hw_decode_ = false;
  AVHWDeviceType hw_device_type_ = AV_HWDEVICE_TYPE_NONE;

  // Video info / geometry.
  int src_w_ = 0;
  int src_h_ = 0;
  int proxy_w_ = 0;
  int proxy_h_ = 0;

  int frame_idx_ = 0;

  VideoInfo info_;
  double fps_hint_ = 30.0;  // For timing / realtime factor.

  // Proxy writer.
  std::unique_ptr<class ProxyVideoWriter> proxy_writer_;
};

}  // namespace airsteady
