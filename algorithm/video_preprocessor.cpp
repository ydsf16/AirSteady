#include "video_preprocessor.h"

#include <algorithm>
#include <stdexcept>

#include <glog/logging.h>

extern "C" {
#include <libavutil/avutil.h>
#include <libavutil/error.h>
#include <libavutil/imgutils.h>
}

namespace airsteady {

namespace {

std::string AvErr2Str(int err) {
  char buf[AV_ERROR_MAX_STRING_SIZE] = {0};
  av_strerror(err, buf, sizeof(buf));
  return std::string(buf);
}

double RationalToDouble(AVRational r) {
  if (r.num == 0 || r.den == 0) return 0.0;
  return static_cast<double>(r.num) / static_cast<double>(r.den);
}

}  // namespace

// ---------------- ProxyVideoWriter (minimal internal helper) ----------------

class ProxyVideoWriter {
 public:
  ProxyVideoWriter() = default;
  ~ProxyVideoWriter() { Close(); }

  void Open(const std::string& path, int w, int h, double fps);
  void Write(const cv::Mat& bgr);
  void Close();

 private:
  AVFormatContext* fmt_ = nullptr;
  AVCodecContext* enc_ctx_ = nullptr;
  AVStream* stream_ = nullptr;

  SwsContext* sws_ = nullptr;
  AVFrame* frame_ = nullptr;
  AVPacket* pkt_ = nullptr;

  int frame_idx_ = 0;
};

void ProxyVideoWriter::Open(const std::string& path, int w, int h, double fps) {
  LOG(INFO) << "[ProxyVideoWriter] Open: path=" << path
            << " size=" << w << "x" << h << " fps=" << fps;

  if (fps <= 0.0) {
    LOG(WARNING) << "[ProxyVideoWriter] fps <= 0, fallback to 30";
    fps = 30.0;
  }

  int ret = avformat_alloc_output_context2(&fmt_, nullptr, nullptr, path.c_str());
  if (ret < 0 || !fmt_) {
    LOG(ERROR) << "[ProxyVideoWriter] avformat_alloc_output_context2 failed: " << AvErr2Str(ret);
    throw std::runtime_error("avformat_alloc_output_context2 failed: " + AvErr2Str(ret));
  }

  const AVCodec* codec = avcodec_find_encoder_by_name("libx264");
  if (!codec) {
    LOG(ERROR) << "[ProxyVideoWriter] libx264 encoder not found";
    throw std::runtime_error("libx264 not found");
  }

  enc_ctx_ = avcodec_alloc_context3(codec);
  if (!enc_ctx_) {
    LOG(ERROR) << "[ProxyVideoWriter] avcodec_alloc_context3 failed";
    throw std::runtime_error("avcodec_alloc_context3 failed");
  }

  enc_ctx_->width = w;
  enc_ctx_->height = h;
  enc_ctx_->pix_fmt = AV_PIX_FMT_YUV420P;
  enc_ctx_->time_base = AVRational{1, static_cast<int>(fps + 0.5)};
  if (enc_ctx_->time_base.den <= 0) enc_ctx_->time_base.den = 30;
  enc_ctx_->framerate = AVRational{static_cast<int>(fps + 0.5), 1};
  enc_ctx_->gop_size = 60;
  enc_ctx_->max_b_frames = 0;

  if (fmt_->oformat->flags & AVFMT_GLOBALHEADER) {
    enc_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  }

  ret = avcodec_open2(enc_ctx_, codec, nullptr);
  if (ret < 0) {
    LOG(ERROR) << "[ProxyVideoWriter] avcodec_open2 failed: " << AvErr2Str(ret);
    avcodec_free_context(&enc_ctx_);
    throw std::runtime_error("avcodec_open2 failed: " + AvErr2Str(ret));
  }

  stream_ = avformat_new_stream(fmt_, nullptr);
  if (!stream_) {
    LOG(ERROR) << "[ProxyVideoWriter] avformat_new_stream failed";
    throw std::runtime_error("avformat_new_stream failed");
  }

  ret = avcodec_parameters_from_context(stream_->codecpar, enc_ctx_);
  if (ret < 0) {
    LOG(ERROR) << "[ProxyVideoWriter] avcodec_parameters_from_context failed: " << AvErr2Str(ret);
    throw std::runtime_error("avcodec_parameters_from_context failed: " + AvErr2Str(ret));
  }
  stream_->time_base = enc_ctx_->time_base;

  if (!(fmt_->oformat->flags & AVFMT_NOFILE)) {
    ret = avio_open(&fmt_->pb, path.c_str(), AVIO_FLAG_WRITE);
    if (ret < 0) {
      LOG(ERROR) << "[ProxyVideoWriter] avio_open failed: " << AvErr2Str(ret);
      throw std::runtime_error("avio_open failed: " + AvErr2Str(ret));
    }
  }

  ret = avformat_write_header(fmt_, nullptr);
  if (ret < 0) {
    LOG(ERROR) << "[ProxyVideoWriter] avformat_write_header failed: " << AvErr2Str(ret);
    throw std::runtime_error("avformat_write_header failed: " + AvErr2Str(ret));
  }

  sws_ = sws_getContext(w, h, AV_PIX_FMT_BGR24,
                        w, h, enc_ctx_->pix_fmt,
                        SWS_BILINEAR, nullptr, nullptr, nullptr);
  if (!sws_) {
    LOG(ERROR) << "[ProxyVideoWriter] sws_getContext failed";
    throw std::runtime_error("sws_getContext failed");
  }

  frame_ = av_frame_alloc();
  if (!frame_) {
    LOG(ERROR) << "[ProxyVideoWriter] av_frame_alloc failed";
    throw std::runtime_error("av_frame_alloc failed");
  }
  frame_->format = enc_ctx_->pix_fmt;
  frame_->width = w;
  frame_->height = h;
  ret = av_frame_get_buffer(frame_, 32);
  if (ret < 0) {
    LOG(ERROR) << "[ProxyVideoWriter] av_frame_get_buffer failed: " << AvErr2Str(ret);
    throw std::runtime_error("av_frame_get_buffer failed: " + AvErr2Str(ret));
  }

  pkt_ = av_packet_alloc();
  if (!pkt_) {
    LOG(ERROR) << "[ProxyVideoWriter] av_packet_alloc failed";
    throw std::runtime_error("av_packet_alloc failed");
  }
}

void ProxyVideoWriter::Write(const cv::Mat& bgr) {
  if (!fmt_ || !enc_ctx_ || !frame_) {
    LOG(ERROR) << "[ProxyVideoWriter] Write called on uninitialized writer";
    return;
  }
  if (bgr.empty()) {
    LOG(WARNING) << "[ProxyVideoWriter] Write called with empty frame";
    return;
  }
  if (bgr.cols != enc_ctx_->width || bgr.rows != enc_ctx_->height || bgr.type() != CV_8UC3) {
    LOG(ERROR) << "[ProxyVideoWriter] Write: unexpected frame size/type. "
               << "got=" << bgr.cols << "x" << bgr.rows
               << " type=" << bgr.type()
               << " expected=" << enc_ctx_->width << "x" << enc_ctx_->height
               << " type=" << CV_8UC3;
    return;
  }

  int ret = av_frame_make_writable(frame_);
  if (ret < 0) {
    LOG(ERROR) << "[ProxyVideoWriter] av_frame_make_writable failed: " << AvErr2Str(ret);
    return;
  }

  const uint8_t* src[4] = {bgr.data, nullptr, nullptr, nullptr};
  int src_stride[4] = {static_cast<int>(bgr.step), 0, 0, 0};

  ret = sws_scale(sws_, src, src_stride, 0, frame_->height,
                  frame_->data, frame_->linesize);
  if (ret <= 0) {
    LOG(ERROR) << "[ProxyVideoWriter] sws_scale failed, ret=" << ret;
    return;
  }

  frame_->pts = frame_idx_++;

  ret = avcodec_send_frame(enc_ctx_, frame_);
  if (ret < 0) {
    LOG(ERROR) << "[ProxyVideoWriter] avcodec_send_frame failed: " << AvErr2Str(ret);
    return;
  }

  while (true) {
    ret = avcodec_receive_packet(enc_ctx_, pkt_);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
    if (ret < 0) {
      LOG(ERROR) << "[ProxyVideoWriter] avcodec_receive_packet failed: " << AvErr2Str(ret);
      break;
    }

    av_packet_rescale_ts(pkt_, enc_ctx_->time_base, stream_->time_base);
    pkt_->stream_index = stream_->index;

    int wret = av_interleaved_write_frame(fmt_, pkt_);
    if (wret < 0) {
      LOG(ERROR) << "[ProxyVideoWriter] av_interleaved_write_frame failed: " << AvErr2Str(wret);
      av_packet_unref(pkt_);
      break;
    }
    av_packet_unref(pkt_);
  }
}

void ProxyVideoWriter::Close() {
  if (!enc_ctx_ || !fmt_) return;

  LOG(INFO) << "[ProxyVideoWriter] Close, total frames=" << frame_idx_;

  int ret = avcodec_send_frame(enc_ctx_, nullptr);
  if (ret < 0 && ret != AVERROR_EOF) {
    LOG(WARNING) << "[ProxyVideoWriter] avcodec_send_frame(flush) failed: " << AvErr2Str(ret);
  }

  if (pkt_) {
    while (true) {
      ret = avcodec_receive_packet(enc_ctx_, pkt_);
      if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
      if (ret < 0) {
        LOG(WARNING) << "[ProxyVideoWriter] avcodec_receive_packet(flush) failed: " << AvErr2Str(ret);
        break;
      }
      av_packet_rescale_ts(pkt_, enc_ctx_->time_base, stream_ ? stream_->time_base : enc_ctx_->time_base);
      if (stream_) pkt_->stream_index = stream_->index;
      int wret = av_interleaved_write_frame(fmt_, pkt_);
      if (wret < 0) {
        LOG(WARNING) << "[ProxyVideoWriter] av_interleaved_write_frame(flush) failed: " << AvErr2Str(wret);
        av_packet_unref(pkt_);
        break;
      }
      av_packet_unref(pkt_);
    }
  }

  ret = av_write_trailer(fmt_);
  if (ret < 0) {
    LOG(WARNING) << "[ProxyVideoWriter] av_write_trailer failed: " << AvErr2Str(ret);
  }

  if (pkt_) av_packet_free(&pkt_);
  if (frame_) av_frame_free(&frame_);
  if (sws_) sws_freeContext(sws_);
  if (enc_ctx_) avcodec_free_context(&enc_ctx_);
  if (fmt_) {
    if (!(fmt_->oformat->flags & AVFMT_NOFILE)) {
      int c = avio_closep(&fmt_->pb);
      if (c < 0) {
        LOG(WARNING) << "[ProxyVideoWriter] avio_closep failed: " << AvErr2Str(c);
      }
    }
    avformat_free_context(fmt_);
  }

  pkt_ = nullptr;
  frame_ = nullptr;
  sws_ = nullptr;
  enc_ctx_ = nullptr;
  fmt_ = nullptr;
  stream_ = nullptr;
  frame_idx_ = 0;
}

// ---------------- VideoPreprocessor ----------------

VideoPreprocessor::VideoPreprocessor(const std::string& src,
                                     const std::string& proxy,
                                     int max_proxy_resolution,
                                     size_t num_prefetch)
    : src_path_(src),
      proxy_path_(proxy),
      max_proxy_res_(max_proxy_resolution),
      max_prefetch_(num_prefetch) {}

VideoPreprocessor::~VideoPreprocessor() {
  LOG(INFO) << "[VideoPreprocessor] Destroy, stopping worker thread";
  stop_ = true;
  cv_not_empty_.notify_all();
  cv_not_full_.notify_all();
  if (worker_.joinable()) worker_.join();

  if (sws_) sws_freeContext(sws_);
  if (frame_) av_frame_free(&frame_);
  if (sw_frame_) av_frame_free(&sw_frame_);
  if (pkt_) av_packet_free(&pkt_);
  if (dec_ctx_) avcodec_free_context(&dec_ctx_);
  if (fmt_) avformat_close_input(&fmt_);
}

bool VideoPreprocessor::TryOpenVideo(std::string* err) {
  LOG(INFO) << "[VideoPreprocessor] TryOpenVideo, src=" << src_path_
            << " proxy=" << proxy_path_
            << " max_proxy_res=" << max_proxy_res_
            << " prefetch=" << max_prefetch_;

  if (!InitFFmpeg(err)) {
    LOG(ERROR) << "[VideoPreprocessor] InitFFmpeg failed";
    return false;
  }

  LOG(INFO) << "[VideoPreprocessor] Source size=" << src_w_ << "x" << src_h_;

  int long_edge = std::max(src_w_, src_h_);
  if (long_edge > max_proxy_res_) {
    double s = static_cast<double>(max_proxy_res_) / long_edge;
    proxy_w_ = static_cast<int>(src_w_ * s + 0.5);
    proxy_h_ = static_cast<int>(src_h_ * s + 0.5);
  } else {
    proxy_w_ = src_w_;
    proxy_h_ = src_h_;
  }

  // Force even dimensions.
  proxy_w_ &= ~1;
  proxy_h_ &= ~1;
  proxy_w_ = std::max(proxy_w_, 2);
  proxy_h_ = std::max(proxy_h_, 2);

  LOG(INFO) << "[VideoPreprocessor] Proxy size=" << proxy_w_ << "x" << proxy_h_;

  // Compute fps for writer (best-effort).
  double fps = 0.0;
  if (info_.total_time_sec > 0.0 && info_.num_frames > 0) {
    fps = static_cast<double>(info_.num_frames) / info_.total_time_sec;
  } else if (video_st_) {
    AVRational fr = video_st_->avg_frame_rate.num && video_st_->avg_frame_rate.den
                        ? video_st_->avg_frame_rate
                        : video_st_->r_frame_rate;
    fps = RationalToDouble(fr);
  }
  if (fps <= 0.0) fps = 30.0;
  LOG(INFO) << "[VideoPreprocessor] Proxy fps=" << fps;

  try {
    proxy_writer_ = std::make_unique<ProxyVideoWriter>();
    proxy_writer_->Open(proxy_path_, proxy_w_, proxy_h_, fps);
  } catch (const std::exception& e) {
    LOG(ERROR) << "[VideoPreprocessor] Failed to open proxy writer: " << e.what();
    if (err) *err = e.what();
    return false;
  }

  worker_ = std::thread(&VideoPreprocessor::PrefetchThread, this);
  LOG(INFO) << "[VideoPreprocessor] Prefetch thread started";
  return true;
}

VideoInfo VideoPreprocessor::GetVideoInfo() const { return info_; }

std::shared_ptr<PreFrame> VideoPreprocessor::NextFrame() {
  std::unique_lock lk(mtx_);
  cv_not_empty_.wait(lk, [&] {
    return !queue_.empty() || eof_ || stop_;
  });

  if (queue_.empty()) {
    if (eof_) {
      LOG(INFO) << "[VideoPreprocessor] NextFrame: EOF and queue empty";
    } else if (stop_) {
      LOG(INFO) << "[VideoPreprocessor] NextFrame: stopped and queue empty";
    }
    return nullptr;
  }

  auto f = queue_.front();
  queue_.pop_front();
  cv_not_full_.notify_one();

  VLOG(2) << "[VideoPreprocessor] NextFrame: idx=" << f->frame_idx
          << " time_ns=" << f->time_ns
          << " remaining_queue=" << queue_.size();

  return f;
}

bool VideoPreprocessor::InitFFmpeg(std::string* err) {
  avformat_network_init();

  int ret = avformat_open_input(&fmt_, src_path_.c_str(), nullptr, nullptr);
  if (ret < 0 || !fmt_) {
    std::string msg = "avformat_open_input failed: " + AvErr2Str(ret);
    LOG(ERROR) << "[VideoPreprocessor] " << msg;
    if (err) *err = msg;
    return false;
  }

  ret = avformat_find_stream_info(fmt_, nullptr);
  if (ret < 0) {
    std::string msg = "avformat_find_stream_info failed: " + AvErr2Str(ret);
    LOG(ERROR) << "[VideoPreprocessor] " << msg;
    if (err) *err = msg;
    return false;
  }

  video_stream_idx_ =
      av_find_best_stream(fmt_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  if (video_stream_idx_ < 0) {
    std::string msg = "av_find_best_stream(video) failed: " + AvErr2Str(video_stream_idx_);
    LOG(ERROR) << "[VideoPreprocessor] " << msg;
    if (err) *err = msg;
    return false;
  }

  video_st_ = fmt_->streams[video_stream_idx_];

  const AVCodec* dec = avcodec_find_decoder(video_st_->codecpar->codec_id);
  if (!dec) {
    std::string msg = "avcodec_find_decoder failed";
    LOG(ERROR) << "[VideoPreprocessor] " << msg;
    if (err) *err = msg;
    return false;
  }

  dec_ctx_ = avcodec_alloc_context3(dec);
  if (!dec_ctx_) {
    std::string msg = "avcodec_alloc_context3 failed";
    LOG(ERROR) << "[VideoPreprocessor] " << msg;
    if (err) *err = msg;
    return false;
  }

  ret = avcodec_parameters_to_context(dec_ctx_, video_st_->codecpar);
  if (ret < 0) {
    std::string msg = "avcodec_parameters_to_context failed: " + AvErr2Str(ret);
    LOG(ERROR) << "[VideoPreprocessor] " << msg;
    if (err) *err = msg;
    return false;
  }

  ret = avcodec_open2(dec_ctx_, dec, nullptr);
  if (ret < 0) {
    std::string msg = "avcodec_open2 failed: " + AvErr2Str(ret);
    LOG(ERROR) << "[VideoPreprocessor] " << msg;
    if (err) *err = msg;
    return false;
  }

  src_w_ = dec_ctx_->width;
  src_h_ = dec_ctx_->height;

  info_.width = src_w_;
  info_.height = src_h_;
  info_.codec = dec->name ? dec->name : "unknown";
  info_.bitrate = fmt_->bit_rate;

  if (fmt_->duration > 0 && fmt_->duration != AV_NOPTS_VALUE) {
    info_.total_time_sec = static_cast<double>(fmt_->duration) / AV_TIME_BASE;
  } else {
    info_.total_time_sec = 0.0;
  }

  info_.num_frames = static_cast<int>(video_st_->nb_frames);

  // Log basic info.
  double fps_guess = 0.0;
  if (video_st_->avg_frame_rate.num && video_st_->avg_frame_rate.den) {
    fps_guess = RationalToDouble(video_st_->avg_frame_rate);
  } else if (video_st_->r_frame_rate.num && video_st_->r_frame_rate.den) {
    fps_guess = RationalToDouble(video_st_->r_frame_rate);
  }

  LOG(INFO) << "[VideoPreprocessor] Opened video: codec=" << info_.codec
            << " size=" << info_.width << "x" << info_.height
            << " bitrate=" << info_.bitrate
            << " duration_sec=" << info_.total_time_sec
            << " nb_frames=" << info_.num_frames
            << " fps_guess=" << fps_guess;

  pkt_ = av_packet_alloc();
  frame_ = av_frame_alloc();
  sw_frame_ = av_frame_alloc();
  if (!pkt_ || !frame_ || !sw_frame_) {
    std::string msg = "av_packet_alloc/av_frame_alloc failed";
    LOG(ERROR) << "[VideoPreprocessor] " << msg;
    if (err) *err = msg;
    return false;
  }

  return true;
}

bool VideoPreprocessor::DecodeOneFrame(AVFrame* out) {
  while (true) {
    int ret = av_read_frame(fmt_, pkt_);
    if (ret == AVERROR_EOF) {
      LOG(INFO) << "[VideoPreprocessor] av_read_frame EOF";
      return false;
    }
    if (ret < 0) {
      LOG(ERROR) << "[VideoPreprocessor] av_read_frame failed: " << AvErr2Str(ret);
      return false;
    }

    if (pkt_->stream_index != video_stream_idx_) {
      av_packet_unref(pkt_);
      continue;
    }

    ret = avcodec_send_packet(dec_ctx_, pkt_);
    av_packet_unref(pkt_);
    if (ret < 0) {
      LOG(ERROR) << "[VideoPreprocessor] avcodec_send_packet failed: " << AvErr2Str(ret);
      continue;
    }

    ret = avcodec_receive_frame(dec_ctx_, out);
    if (ret == AVERROR(EAGAIN)) {
      // Need more packets.
      continue;
    }
    if (ret < 0) {
      LOG(ERROR) << "[VideoPreprocessor] avcodec_receive_frame failed: " << AvErr2Str(ret);
      return false;
    }

    VLOG(3) << "[VideoPreprocessor] Decoded frame: pts=" << out->pts
            << " best_effort_ts=" << out->best_effort_timestamp
            << " size=" << out->width << "x" << out->height;
    return true;
  }
}

void VideoPreprocessor::PrefetchThread() {
  LOG(INFO) << "[VideoPreprocessor] PrefetchThread started";

  while (!stop_) {
    {
      std::unique_lock lk(mtx_);
      cv_not_full_.wait(lk, [&] {
        return stop_ || queue_.size() < max_prefetch_;
      });
    }

    if (stop_) {
      LOG(INFO) << "[VideoPreprocessor] PrefetchThread: stop_ set, exiting";
      break;
    }

    if (!DecodeOneFrame(frame_)) {
      eof_ = true;
      LOG(INFO) << "[VideoPreprocessor] PrefetchThread: DecodeOneFrame returned false, EOF or error";
      cv_not_empty_.notify_all();
      break;
    }

    if (!sws_) {
      sws_ = sws_getContext(frame_->width, frame_->height,
                            static_cast<AVPixelFormat>(frame_->format),
                            proxy_w_, proxy_h_,
                            AV_PIX_FMT_BGR24,
                            SWS_BILINEAR, nullptr, nullptr, nullptr);
      if (!sws_) {
        LOG(ERROR) << "[VideoPreprocessor] sws_getContext failed, stopping prefetch thread";
        eof_ = true;
        cv_not_empty_.notify_all();
        break;
      }
      LOG(INFO) << "[VideoPreprocessor] sws context created: src_fmt="
                << av_get_pix_fmt_name(static_cast<AVPixelFormat>(frame_->format))
                << " src_size=" << frame_->width << "x" << frame_->height
                << " dst_size=" << proxy_w_ << "x" << proxy_h_;
    }

    cv::Mat bgr(proxy_h_, proxy_w_, CV_8UC3);
    uint8_t* dst[4] = {bgr.data, nullptr, nullptr, nullptr};
    int dst_stride[4] = {static_cast<int>(bgr.step), 0, 0, 0};

    int ret = sws_scale(sws_, frame_->data, frame_->linesize,
                        0, frame_->height, dst, dst_stride);
    if (ret <= 0) {
      LOG(ERROR) << "[VideoPreprocessor] sws_scale failed in PrefetchThread, ret=" << ret;
      eof_ = true;
      cv_not_empty_.notify_all();
      break;
    }

    if (proxy_writer_) {
      proxy_writer_->Write(bgr);
    }

    auto pf = std::make_shared<PreFrame>();
    pf->frame_idx = frame_idx_++;
    pf->proxy_bgr = bgr;
    cv::cvtColor(bgr, pf->proxy_gray, cv::COLOR_BGR2GRAY);

    int64_t pts = (frame_->best_effort_timestamp != AV_NOPTS_VALUE)
                      ? frame_->best_effort_timestamp
                      : frame_->pts;
    if (pts == AV_NOPTS_VALUE) pts = 0;

    double t_sec = pts * RationalToDouble(video_st_->time_base);
    pf->time_ns = static_cast<int64_t>(t_sec * 1e9);

    {
      std::lock_guard lk(mtx_);
      queue_.push_back(pf);
      VLOG(2) << "[VideoPreprocessor] Prefetched frame idx=" << pf->frame_idx
              << " time_ns=" << pf->time_ns
              << " queue_size=" << queue_.size();
    }
    cv_not_empty_.notify_one();

    av_frame_unref(frame_);
  }

  LOG(INFO) << "[VideoPreprocessor] PrefetchThread exit, total_frames=" << frame_idx_;
}

}  // namespace airsteady
