#include "video_exportor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <glog/logging.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/error.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/hwcontext.h>
#include <libswscale/swscale.h>
#include <libavutil/channel_layout.h>

// NEW: audio transcode
#include <libswresample/swresample.h>
#include <libavutil/audio_fifo.h>
}

namespace airsteady {
namespace {

using Clock = std::chrono::steady_clock;

static void FixupAudioChannelLayout(AVCodecParameters* par) {
  if (!par) return;

  if (par->codec_type != AVMEDIA_TYPE_AUDIO) return;

  if (par->ch_layout.nb_channels > 0) {
    if (par->ch_layout.order == AV_CHANNEL_ORDER_UNSPEC) {
      AVChannelLayout def{};
      av_channel_layout_default(&def, par->ch_layout.nb_channels);
      av_channel_layout_uninit(&par->ch_layout);
      par->ch_layout = def;
    }
    return;
  }

#if FF_API_OLD_CHANNEL_LAYOUT
  if (par->channels > 0) {
    AVChannelLayout def{};
    av_channel_layout_default(&def, par->channels);
    av_channel_layout_uninit(&par->ch_layout);
    par->ch_layout = def;
  }
#endif
}

static void FixupAudioChannelLayoutCtx(AVCodecContext* ctx) {
  if (!ctx) return;
  if (ctx->codec_type != AVMEDIA_TYPE_AUDIO) return;
  if (ctx->ch_layout.nb_channels <= 0) {
    // Best effort: default to stereo.
    AVChannelLayout def{};
    av_channel_layout_default(&def, 2);
    av_channel_layout_uninit(&ctx->ch_layout);
    ctx->ch_layout = def;
    return;
  }
  if (ctx->ch_layout.order == AV_CHANNEL_ORDER_UNSPEC) {
    AVChannelLayout def{};
    av_channel_layout_default(&def, ctx->ch_layout.nb_channels);
    av_channel_layout_uninit(&ctx->ch_layout);
    ctx->ch_layout = def;
  }
}

double MsSince(const Clock::time_point& t0, const Clock::time_point& t1) {
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

std::string AvErr2Str(int err) {
  char buf[AV_ERROR_MAX_STRING_SIZE] = {0};
  av_strerror(err, buf, sizeof(buf));
  return std::string(buf);
}

double RationalToDouble(AVRational r) {
  if (r.num == 0 || r.den == 0) return 0.0;
  return static_cast<double>(r.num) / static_cast<double>(r.den);
}

int GetPixFmtBitDepth(AVPixelFormat fmt) {
  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(fmt);
  if (!desc) return 8;
  return desc->comp[0].depth > 0 ? desc->comp[0].depth : 8;
}

bool IsHwFrameFormat(AVPixelFormat fmt) {
  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(fmt);
  return desc && (desc->flags & AV_PIX_FMT_FLAG_HWACCEL);
}

bool CodecSupportsPixFmt(const AVCodec* codec, AVPixelFormat fmt) {
  if (!codec) return false;
  if (!codec->pix_fmts) return true;  // unknown -> try open2
  for (const AVPixelFormat* p = codec->pix_fmts; *p != AV_PIX_FMT_NONE; ++p) {
    if (*p == fmt) return true;
  }
  return false;
}

int64_t FrameBytesForPixFmt(int w, int h, AVPixelFormat fmt) {
  if (w <= 0 || h <= 0) return 0;
  if (fmt == AV_PIX_FMT_YUV420P) return static_cast<int64_t>(w) * h * 3 / 2;
  if (fmt == AV_PIX_FMT_P010LE) return static_cast<int64_t>(w) * h * 3;
  if (fmt == AV_PIX_FMT_YUV420P10LE) return static_cast<int64_t>(w) * h * 3;
  int size = av_image_get_buffer_size(fmt, w, h, 1);
  if (size > 0) return size;
  return static_cast<int64_t>(w) * h * 3 / 2;
}

bool CopyFileBinary(const std::string& src, const std::string& dst, std::string* err) {
  std::ifstream in(src, std::ios::binary);
  if (!in) {
    if (err) *err = "CopyFileBinary: open src failed: " + src;
    return false;
  }
  std::ofstream out(dst, std::ios::binary | std::ios::trunc);
  if (!out) {
    if (err) *err = "CopyFileBinary: open dst failed: " + dst;
    return false;
  }
  out << in.rdbuf();
  if (!out.good()) {
    if (err) *err = "CopyFileBinary: write failed: " + dst;
    return false;
  }
  return true;
}

bool ReplaceFileByRenameOrCopy(const std::string& src, const std::string& dst, std::string* err) {
  std::remove(dst.c_str());
  if (std::rename(src.c_str(), dst.c_str()) == 0) {
    return true;
  }
  if (!CopyFileBinary(src, dst, err)) return false;
  std::remove(src.c_str());
  return true;
}

// Try create HW device (best-effort).
AVBufferRef* TryCreateHwDeviceCtx(bool verbose, AVHWDeviceType* out_type) {
  const AVHWDeviceType pref[] = {
      AV_HWDEVICE_TYPE_D3D11VA,
      AV_HWDEVICE_TYPE_DXVA2,
      AV_HWDEVICE_TYPE_CUDA,
      AV_HWDEVICE_TYPE_QSV,
      AV_HWDEVICE_TYPE_VAAPI,
      AV_HWDEVICE_TYPE_VIDEOTOOLBOX,
  };

  if (out_type) *out_type = AV_HWDEVICE_TYPE_NONE;

  for (AVHWDeviceType t : pref) {
    if (t == AV_HWDEVICE_TYPE_NONE) continue;
    AVBufferRef* dev = nullptr;
    int ret = av_hwdevice_ctx_create(&dev, t, nullptr, nullptr, 0);
    if (ret == 0 && dev) {
      if (out_type) *out_type = t;
      if (verbose) {
        LOG(INFO) << "[VideoExportor] Using HW device: "
                  << av_hwdevice_get_type_name(t);
      }
      return dev;
    }
    if (verbose) {
      LOG(INFO) << "[VideoExportor] HW device "
                << av_hwdevice_get_type_name(t)
                << " not available: " << AvErr2Str(ret);
    }
  }

  if (verbose) {
    LOG(INFO) << "[VideoExportor] No HW device available; using SW decode.";
  }
  return nullptr;
}

// ---------------- ExportVideoWriter ----------------

class ExportVideoWriter {
 public:
  ExportVideoWriter() = default;
  ~ExportVideoWriter() { Close(); }

  void Open(const std::string& path,
            int out_w,
            int out_h,
            AVRational fps_q,
            int64_t bitrate_bps,
            AVPixelFormat preferred_pix_fmt,
            const std::vector<const char*>& encoder_names);

  void WriteFrame(AVFrame* frame);
  void Close();

  const std::string& encoder_name() const { return encoder_name_; }
  AVPixelFormat pix_fmt() const { return enc_ctx_ ? enc_ctx_->pix_fmt : AV_PIX_FMT_NONE; }
  AVRational time_base() const { return enc_ctx_ ? enc_ctx_->time_base : AVRational{1, 30}; }

 private:
  AVFormatContext* fmt_ = nullptr;
  AVCodecContext* enc_ctx_ = nullptr;
  AVStream* stream_ = nullptr;

  AVPacket* pkt_ = nullptr;
  std::string encoder_name_;
  int64_t frame_idx_ = 0;
  bool header_written_ = false;
};

void ExportVideoWriter::Open(const std::string& path,
                             int out_w,
                             int out_h,
                             AVRational fps_q,
                             int64_t bitrate_bps,
                             AVPixelFormat preferred_pix_fmt,
                             const std::vector<const char*>& encoder_names) {
  if (fps_q.num <= 0 || fps_q.den <= 0) fps_q = AVRational{30, 1};
  header_written_ = false;

  LOG(INFO) << "[ExportVideoWriter] Open: path=" << path
            << " size=" << out_w << "x" << out_h
            << " fps_q=" << fps_q.num << "/" << fps_q.den
            << " bitrate=" << bitrate_bps
            << " preferred_pix_fmt=" << av_get_pix_fmt_name(preferred_pix_fmt);

  if (out_w <= 0 || out_h <= 0) {
    throw std::runtime_error("ExportVideoWriter::Open invalid size");
  }

  if (out_w & 1) ++out_w;
  if (out_h & 1) ++out_h;

  int ret = avformat_alloc_output_context2(&fmt_, nullptr, nullptr, path.c_str());
  if (ret < 0 || !fmt_) {
    throw std::runtime_error("avformat_alloc_output_context2 failed: " + AvErr2Str(ret));
  }

  stream_ = avformat_new_stream(fmt_, nullptr);
  if (!stream_) {
    throw std::runtime_error("avformat_new_stream failed");
  }

  bool opened = false;

  for (const char* name : encoder_names) {
    const AVCodec* codec = avcodec_find_encoder_by_name(name);
    if (!codec) {
      LOG(INFO) << "[ExportVideoWriter] Encoder not found: " << name;
      continue;
    }

    std::vector<AVPixelFormat> try_fmts;
    try_fmts.push_back(preferred_pix_fmt);
    if (preferred_pix_fmt == AV_PIX_FMT_P010LE) {
      try_fmts.push_back(AV_PIX_FMT_YUV420P10LE);
    } else if (preferred_pix_fmt == AV_PIX_FMT_YUV420P10LE) {
      try_fmts.push_back(AV_PIX_FMT_P010LE);
    }
    try_fmts.erase(std::unique(try_fmts.begin(), try_fmts.end()), try_fmts.end());

    for (AVPixelFormat fmt_try : try_fmts) {
      if (!CodecSupportsPixFmt(codec, fmt_try)) {
        LOG(INFO) << "[ExportVideoWriter] Encoder " << name
                  << " does not list pix_fmt=" << av_get_pix_fmt_name(fmt_try);
        continue;
      }

      AVCodecContext* ctx = avcodec_alloc_context3(codec);
      if (!ctx) {
        LOG(WARNING) << "[ExportVideoWriter] avcodec_alloc_context3 failed for " << name;
        continue;
      }

      ctx->width = out_w;
      ctx->height = out_h;

      ctx->framerate = fps_q;
      ctx->time_base = av_inv_q(fps_q);

      ctx->gop_size = 60;
      ctx->max_b_frames = 0;
      ctx->pix_fmt = fmt_try;

      if (bitrate_bps > 0) ctx->bit_rate = bitrate_bps;

      if (fmt_->oformat->flags & AVFMT_GLOBALHEADER) {
        ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
      }

      AVDictionary* opt = nullptr;

      if (std::string(name) == "libx264") {
        av_dict_set(&opt, "preset", "veryfast", 0);
        av_dict_set(&opt, "tune", "zerolatency", 0);
      }
      if (std::string(name) == "libx265") {
        av_dict_set(&opt, "preset", "fast", 0);
        av_dict_set(&opt, "x265-params", "profile=main10", 0);
      }
      if (GetPixFmtBitDepth(fmt_try) > 8) {
        if (std::string(name).find("hevc_") == 0 ||
            std::string(name).find("_nvenc") != std::string::npos) {
          av_dict_set(&opt, "profile", "main10", 0);
        }
      }

      ret = avcodec_open2(ctx, codec, opt ? &opt : nullptr);
      if (opt) av_dict_free(&opt);

      if (ret < 0) {
        LOG(WARNING) << "[ExportVideoWriter] avcodec_open2 failed for encoder "
                     << name << " pix_fmt=" << av_get_pix_fmt_name(fmt_try)
                     << " err=" << AvErr2Str(ret);
        avcodec_free_context(&ctx);
        continue;
      }

      enc_ctx_ = ctx;
      encoder_name_ = codec->name ? codec->name : "unknown";
      opened = true;
      LOG(INFO) << "[ExportVideoWriter] Selected encoder=" << encoder_name_
                << " pix_fmt=" << av_get_pix_fmt_name(enc_ctx_->pix_fmt)
                << " depth=" << GetPixFmtBitDepth(enc_ctx_->pix_fmt);
      break;
    }

    if (opened) break;
  }

  if (!opened || !enc_ctx_) {
    throw std::runtime_error("No usable encoder found for export.");
  }

  stream_->time_base = enc_ctx_->time_base;
  stream_->avg_frame_rate = enc_ctx_->framerate;
  stream_->r_frame_rate = enc_ctx_->framerate;

  ret = avcodec_parameters_from_context(stream_->codecpar, enc_ctx_);
  if (ret < 0) {
    throw std::runtime_error("avcodec_parameters_from_context failed: " + AvErr2Str(ret));
  }

  if (!(fmt_->oformat->flags & AVFMT_NOFILE)) {
    ret = avio_open(&fmt_->pb, path.c_str(), AVIO_FLAG_WRITE);
    if (ret < 0) {
      throw std::runtime_error("avio_open failed: " + AvErr2Str(ret));
    }
  }

  ret = avformat_write_header(fmt_, nullptr);
  if (ret < 0) {
    throw std::runtime_error("avformat_write_header failed: " + AvErr2Str(ret));
  }
  header_written_ = true;

  pkt_ = av_packet_alloc();
  if (!pkt_) {
    throw std::runtime_error("av_packet_alloc failed");
  }

  frame_idx_ = 0;
}

void ExportVideoWriter::WriteFrame(AVFrame* frame) {
  if (!fmt_ || !enc_ctx_ || !stream_ || !pkt_ || !frame) return;

  if (frame->format != enc_ctx_->pix_fmt ||
      frame->width != enc_ctx_->width ||
      frame->height != enc_ctx_->height) {
    LOG(ERROR) << "[ExportVideoWriter] WriteFrame unexpected format/size. got fmt="
               << av_get_pix_fmt_name(static_cast<AVPixelFormat>(frame->format))
               << " expect fmt=" << av_get_pix_fmt_name(enc_ctx_->pix_fmt)
               << " got size=" << frame->width << "x" << frame->height
               << " expect size=" << enc_ctx_->width << "x" << enc_ctx_->height;
    return;
  }

  if (frame->pts == AV_NOPTS_VALUE) {
    frame->pts = frame_idx_;
  }
  frame_idx_++;

  int ret = avcodec_send_frame(enc_ctx_, frame);
  if (ret < 0) {
    LOG(ERROR) << "[ExportVideoWriter] avcodec_send_frame failed: " << AvErr2Str(ret);
    return;
  }

  while (true) {
    ret = avcodec_receive_packet(enc_ctx_, pkt_);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
    if (ret < 0) {
      LOG(ERROR) << "[ExportVideoWriter] avcodec_receive_packet failed: " << AvErr2Str(ret);
      break;
    }

    av_packet_rescale_ts(pkt_, enc_ctx_->time_base, stream_->time_base);
    pkt_->stream_index = stream_->index;

    int wret = av_interleaved_write_frame(fmt_, pkt_);
    if (wret < 0) {
      LOG(ERROR) << "[ExportVideoWriter] av_interleaved_write_frame failed: " << AvErr2Str(wret);
      av_packet_unref(pkt_);
      break;
    }
    av_packet_unref(pkt_);
  }
}

void ExportVideoWriter::Close() {
  if (!fmt_ && !enc_ctx_ && !pkt_) return;

  LOG(INFO) << "[ExportVideoWriter] Close, encoder=" << encoder_name_
            << " pix_fmt=" << (enc_ctx_ ? av_get_pix_fmt_name(enc_ctx_->pix_fmt) : "null")
            << " frames=" << frame_idx_
            << " header_written=" << (header_written_ ? 1 : 0);

  if (enc_ctx_) {
    int ret = avcodec_send_frame(enc_ctx_, nullptr);
    if (ret < 0 && ret != AVERROR_EOF) {
      LOG(WARNING) << "[ExportVideoWriter] avcodec_send_frame(flush) failed: " << AvErr2Str(ret);
    }

    if (pkt_) {
      while (true) {
        ret = avcodec_receive_packet(enc_ctx_, pkt_);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
        if (ret < 0) {
          LOG(WARNING) << "[ExportVideoWriter] avcodec_receive_packet(flush) failed: " << AvErr2Str(ret);
          break;
        }

        if (stream_) {
          av_packet_rescale_ts(pkt_, enc_ctx_->time_base, stream_->time_base);
          pkt_->stream_index = stream_->index;
        }

        int wret = av_interleaved_write_frame(fmt_, pkt_);
        if (wret < 0) {
          LOG(WARNING) << "[ExportVideoWriter] av_interleaved_write_frame(flush) failed: " << AvErr2Str(wret);
          av_packet_unref(pkt_);
          break;
        }
        av_packet_unref(pkt_);
      }
    }
  }

  if (fmt_ && header_written_) {
    int tret = av_write_trailer(fmt_);
    if (tret < 0) {
      LOG(WARNING) << "[ExportVideoWriter] av_write_trailer failed: " << AvErr2Str(tret);
    }
  }

  if (pkt_) av_packet_free(&pkt_);
  if (enc_ctx_) avcodec_free_context(&enc_ctx_);

  if (fmt_) {
    if (!(fmt_->oformat->flags & AVFMT_NOFILE)) {
      int c = 0;
      if (fmt_->pb) c = avio_closep(&fmt_->pb);
      if (c < 0) {
        LOG(WARNING) << "[ExportVideoWriter] avio_closep failed: " << AvErr2Str(c);
      }
    }
    avformat_free_context(fmt_);
  }

  pkt_ = nullptr;
  enc_ctx_ = nullptr;
  fmt_ = nullptr;
  stream_ = nullptr;
  encoder_name_.clear();
  frame_idx_ = 0;
  header_written_ = false;
}

// ---------------- ExportDecoder (HW decode preferred) ----------------

class ExportDecoder {
 public:
  ExportDecoder() = default;
  ~ExportDecoder() { Close(); }

  bool Open(const std::string& path, VideoInfo* out_info, std::string* err);
  bool DecodeOneFrame(AVFrame** out_frame);  // false on EOF/error
  void Close();

  int src_width() const { return src_w_; }
  int src_height() const { return src_h_; }
  double fps_hint() const { return fps_hint_; }
  AVRational fps_q() const { return fps_q_; }

 private:
  static enum AVPixelFormat GetHwFormat(AVCodecContext* ctx,
                                       const enum AVPixelFormat* pix_fmts);

 private:
  AVFormatContext* fmt_ = nullptr;
  AVCodecContext* dec_ctx_ = nullptr;
  AVStream* video_st_ = nullptr;
  int video_stream_idx_ = -1;

  AVPacket* pkt_ = nullptr;
  AVFrame* frame_ = nullptr;
  AVFrame* sw_frame_ = nullptr;

  bool using_hw_decode_ = false;
  AVPixelFormat hw_pix_fmt_ = AV_PIX_FMT_NONE;
  AVHWDeviceType hw_device_type_ = AV_HWDEVICE_TYPE_NONE;
  AVBufferRef* hw_device_ctx_ = nullptr;

  int src_w_ = 0;
  int src_h_ = 0;
  double fps_hint_ = 30.0;
  AVRational fps_q_{30, 1};
};

enum AVPixelFormat ExportDecoder::GetHwFormat(AVCodecContext* ctx,
                                             const enum AVPixelFormat* pix_fmts) {
  auto* self = reinterpret_cast<ExportDecoder*>(ctx->opaque);
  for (const enum AVPixelFormat* p = pix_fmts; *p != AV_PIX_FMT_NONE; ++p) {
    if (*p == self->hw_pix_fmt_) {
      LOG(INFO) << "[VideoExportor] GetHwFormat selecting HW pix_fmt="
                << av_get_pix_fmt_name(*p);
      return *p;
    }
  }
  LOG(INFO) << "[VideoExportor] GetHwFormat fallback to "
            << av_get_pix_fmt_name(pix_fmts[0]);
  return pix_fmts[0];
}

bool ExportDecoder::Open(const std::string& path,
                         VideoInfo* out_info,
                         std::string* err) {
  avformat_network_init();

  int ret = avformat_open_input(&fmt_, path.c_str(), nullptr, nullptr);
  if (ret < 0 || !fmt_) {
    if (err) *err = "avformat_open_input failed: " + AvErr2Str(ret);
    return false;
  }

  ret = avformat_find_stream_info(fmt_, nullptr);
  if (ret < 0) {
    if (err) *err = "avformat_find_stream_info failed: " + AvErr2Str(ret);
    return false;
  }

  video_stream_idx_ =
      av_find_best_stream(fmt_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  if (video_stream_idx_ < 0) {
    if (err) *err = "av_find_best_stream(video) failed: " + AvErr2Str(video_stream_idx_);
    return false;
  }

  video_st_ = fmt_->streams[video_stream_idx_];

  const AVCodec* dec = avcodec_find_decoder(video_st_->codecpar->codec_id);
  if (!dec) {
    if (err) *err = "avcodec_find_decoder failed";
    return false;
  }

  dec_ctx_ = avcodec_alloc_context3(dec);
  if (!dec_ctx_) {
    if (err) *err = "avcodec_alloc_context3 failed";
    return false;
  }

  ret = avcodec_parameters_to_context(dec_ctx_, video_st_->codecpar);
  if (ret < 0) {
    if (err) *err = "avcodec_parameters_to_context failed: " + AvErr2Str(ret);
    return false;
  }

  using_hw_decode_ = false;
  hw_pix_fmt_ = AV_PIX_FMT_NONE;
  hw_device_type_ = AV_HWDEVICE_TYPE_NONE;

  hw_device_ctx_ = TryCreateHwDeviceCtx(/*verbose=*/true, &hw_device_type_);
  if (hw_device_ctx_) {
    for (int i = 0;; ++i) {
      const AVCodecHWConfig* cfg = avcodec_get_hw_config(dec, i);
      if (!cfg) break;
      if ((cfg->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) &&
          cfg->device_type == hw_device_type_) {
        hw_pix_fmt_ = cfg->pix_fmt;
        break;
      }
    }

    if (hw_pix_fmt_ != AV_PIX_FMT_NONE) {
      dec_ctx_->opaque = this;
      dec_ctx_->get_format = &ExportDecoder::GetHwFormat;
      dec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
      LOG(INFO) << "[VideoExportor] HW decode enabled: device="
                << av_hwdevice_get_type_name(hw_device_type_)
                << " hw_pix_fmt=" << av_get_pix_fmt_name(hw_pix_fmt_);
    } else {
      LOG(WARNING) << "[VideoExportor] HW device created but no matching HW config; fallback to SW.";
      av_buffer_unref(&hw_device_ctx_);
      hw_device_ctx_ = nullptr;
    }
  }

  ret = avcodec_open2(dec_ctx_, dec, nullptr);
  if (ret < 0) {
    if (err) *err = "avcodec_open2 failed: " + AvErr2Str(ret);
    return false;
  }

  using_hw_decode_ = (hw_device_ctx_ && hw_pix_fmt_ != AV_PIX_FMT_NONE);

  src_w_ = dec_ctx_->width;
  src_h_ = dec_ctx_->height;

  fps_q_ = AVRational{0, 1};
  if (video_st_->avg_frame_rate.num > 0 && video_st_->avg_frame_rate.den > 0) {
    fps_q_ = video_st_->avg_frame_rate;
  } else if (video_st_->r_frame_rate.num > 0 && video_st_->r_frame_rate.den > 0) {
    fps_q_ = video_st_->r_frame_rate;
  } else {
    fps_q_ = AVRational{30, 1};
  }
  fps_hint_ = RationalToDouble(fps_q_);

  pkt_ = av_packet_alloc();
  frame_ = av_frame_alloc();
  sw_frame_ = av_frame_alloc();
  if (!pkt_ || !frame_ || !sw_frame_) {
    if (err) *err = "av_packet_alloc/av_frame_alloc failed";
    return false;
  }

  if (out_info) {
    out_info->width = src_w_;
    out_info->height = src_h_;
    out_info->codec = dec->name ? dec->name : "unknown";
    out_info->bitrate = static_cast<double>(fmt_->bit_rate);
    out_info->fps = fps_hint_;
    if (fmt_->duration > 0 && fmt_->duration != AV_NOPTS_VALUE) {
      out_info->total_time_sec =
          static_cast<double>(fmt_->duration) / AV_TIME_BASE;
    }
    out_info->num_frames = static_cast<int>(video_st_->nb_frames);
  }

  LOG(INFO) << "[VideoExportor] Opened source video: size="
            << src_w_ << "x" << src_h_
            << " fps_q=" << fps_q_.num << "/" << fps_q_.den
            << " using_hw_decode=" << (using_hw_decode_ ? 1 : 0);

  return true;
}

// FIX #1: Do not drop frames; drain decoder fully; flush at EOF.
bool ExportDecoder::DecodeOneFrame(AVFrame** out_frame) {
  if (!fmt_ || !dec_ctx_ || !pkt_ || !frame_ || !sw_frame_) return false;

  while (true) {
    int ret = avcodec_receive_frame(dec_ctx_, frame_);
    if (ret == 0) {
      AVFrame* src = frame_;
      const AVPixelFormat fmt = static_cast<AVPixelFormat>(frame_->format);
      const bool is_hw_frame = IsHwFrameFormat(fmt);

      if (using_hw_decode_ && is_hw_frame) {
        av_frame_unref(sw_frame_);
        int tret = av_hwframe_transfer_data(sw_frame_, frame_, 0);
        if (tret < 0) {
          LOG(ERROR) << "[VideoExportor] av_hwframe_transfer_data failed: " << AvErr2Str(tret);
          return false;
        }
        src = sw_frame_;
      }

      *out_frame = src;
      return true;
    }

    if (ret == AVERROR_EOF) {
      return false;
    }

    if (ret != AVERROR(EAGAIN)) {
      LOG(ERROR) << "[VideoExportor] avcodec_receive_frame failed: " << AvErr2Str(ret);
      return false;
    }

    ret = av_read_frame(fmt_, pkt_);
    if (ret == AVERROR_EOF) {
      int sret = avcodec_send_packet(dec_ctx_, nullptr);
      if (sret < 0 && sret != AVERROR_EOF) {
        LOG(ERROR) << "[VideoExportor] avcodec_send_packet(flush) failed: " << AvErr2Str(sret);
        return false;
      }
      continue;
    }

    if (ret < 0) {
      LOG(ERROR) << "[VideoExportor] av_read_frame failed: " << AvErr2Str(ret);
      return false;
    }

    if (pkt_->stream_index != video_stream_idx_) {
      av_packet_unref(pkt_);
      continue;
    }

    ret = avcodec_send_packet(dec_ctx_, pkt_);
    av_packet_unref(pkt_);
    if (ret < 0) {
      LOG(ERROR) << "[VideoExportor] avcodec_send_packet failed: " << AvErr2Str(ret);
      return false;
    }
  }
}

void ExportDecoder::Close() {
  if (pkt_) av_packet_free(&pkt_);
  if (frame_) av_frame_free(&frame_);
  if (sw_frame_) av_frame_free(&sw_frame_);
  if (dec_ctx_) avcodec_free_context(&dec_ctx_);
  if (fmt_) avformat_close_input(&fmt_);
  if (hw_device_ctx_) av_buffer_unref(&hw_device_ctx_);

  fmt_ = nullptr;
  dec_ctx_ = nullptr;
  video_st_ = nullptr;
  video_stream_idx_ = -1;
  pkt_ = nullptr;
  frame_ = nullptr;
  sw_frame_ = nullptr;
  hw_device_ctx_ = nullptr;
  using_hw_decode_ = false;
  hw_pix_fmt_ = AV_PIX_FMT_NONE;
  hw_device_type_ = AV_HWDEVICE_TYPE_NONE;
  src_w_ = 0;
  src_h_ = 0;
  fps_hint_ = 30.0;
  fps_q_ = AVRational{30, 1};
}

// ---------------- helpers for canvas copy ----------------

struct CanvasPlan {
  int canvas_w = 0;
  int canvas_h = 0;
  int origin_x = 0;
  int origin_y = 0;
};

CanvasPlan ComputeCanvasPlanAndShifts(
    const VideoInfo& info,
    int src_w,
    int src_h,
    const std::vector<FrameStableResult>& stable_results,
    int* out_num_frames,
    std::vector<double>* out_shift_x,
    std::vector<double>* out_shift_y) {
  int nframes = info.num_frames;
  if (nframes <= 0) {
    int max_idx = -1;
    for (const auto& r : stable_results) {
      max_idx = std::max(max_idx, r.frame_idx);
    }
    if (max_idx >= 0) nframes = max_idx + 1;
  }
  if (nframes <= 0) nframes = 1;

  out_shift_x->assign(nframes, 0.0);
  out_shift_y->assign(nframes, 0.0);

  double scale_x = 1.0;
  double scale_y = 1.0;
  if (info.proxy_width > 0) {
    scale_x = static_cast<double>(info.width) /
              static_cast<double>(info.proxy_width);
  }
  if (info.proxy_height > 0) {
    scale_y = static_cast<double>(info.height) /
              static_cast<double>(info.proxy_height);
  }

  for (const auto& r : stable_results) {
    if (r.frame_idx < 0 || r.frame_idx >= nframes) continue;

    // Keep sub-pixel shift in original-resolution pixel units.
    // NOTE: we do NOT quantize to integer pixels here.
    const double dx_orig = -static_cast<double>(r.delta_x) * scale_x;
    const double dy_orig = -static_cast<double>(r.delta_y) * scale_y;

    (*out_shift_x)[r.frame_idx] = dx_orig;
    (*out_shift_y)[r.frame_idx] = dy_orig;
  }

  double min_sx = 0.0, max_sx = 0.0;
  double min_sy = 0.0, max_sy = 0.0;

  if (nframes > 0) {
    min_sx = max_sx = (*out_shift_x)[0];
    min_sy = max_sy = (*out_shift_y)[0];
    for (int i = 1; i < nframes; ++i) {
      min_sx = std::min(min_sx, (*out_shift_x)[i]);
      max_sx = std::max(max_sx, (*out_shift_x)[i]);
      min_sy = std::min(min_sy, (*out_shift_y)[i]);
      max_sy = std::max(max_sy, (*out_shift_y)[i]);
    }
  }

  // Canvas bounds: convert float shift range into integer padding.
  // We bias outward (floor/ceil) to guarantee all frames fit.
  const int min_sx_i = static_cast<int>(std::floor(min_sx));
  const int max_sx_i = static_cast<int>(std::ceil(max_sx));
  const int min_sy_i = static_cast<int>(std::floor(min_sy));
  const int max_sy_i = static_cast<int>(std::ceil(max_sy));

  int canvas_w = src_w + (max_sx_i - min_sx_i);
  int canvas_h = src_h + (max_sy_i - min_sy_i);
  if (canvas_w <= 0) canvas_w = 2;
  if (canvas_h <= 0) canvas_h = 2;

  // Final output resolution must be even.
  if (canvas_w & 1) ++canvas_w;
  if (canvas_h & 1) ++canvas_h;

  CanvasPlan plan;
  plan.canvas_w = canvas_w;
  plan.canvas_h = canvas_h;
  plan.origin_x = -min_sx_i;
  plan.origin_y = -min_sy_i;

  *out_num_frames = nframes;
  return plan;
}


struct CopyRect {
  int src_x = 0;
  int src_y = 0;
  int dst_x = 0;
  int dst_y = 0;
  int w = 0;
  int h = 0;
};

CopyRect ComputeCopyRect(int src_w,
                         int src_h,
                         int dst_w,
                         int dst_h,
                         int dst_x,
                         int dst_y) {
  CopyRect r;
  int sx = 0, sy = 0;
  int dx = dst_x, dy = dst_y;
  int cw = src_w;
  int ch = src_h;

  if (dx < 0) {
    int shift = -dx;
    sx += shift;
    cw -= shift;
    dx = 0;
  }
  if (dy < 0) {
    int shift = -dy;
    sy += shift;
    ch -= shift;
    dy = 0;
  }
  if (dx + cw > dst_w) {
    int shift = dx + cw - dst_w;
    cw -= shift;
  }
  if (dy + ch > dst_h) {
    int shift = dy + ch - dst_h;
    ch -= shift;
  }

  if (cw <= 0 || ch <= 0) {
    r.w = r.h = 0;
    return r;
  }

  r.src_x = sx;
  r.src_y = sy;
  r.dst_x = dx;
  r.dst_y = dy;
  r.w = cw;
  r.h = ch;
  return r;
}

void ClearYuv420pFrame(AVFrame* f) {
  if (!f || f->format != AV_PIX_FMT_YUV420P) return;
  const int w = f->width;
  const int h = f->height;

  for (int y = 0; y < h; ++y) {
    std::memset(f->data[0] + y * f->linesize[0], 16, w);
  }
  const int cw = w / 2;
  const int ch = h / 2;
  for (int y = 0; y < ch; ++y) {
    std::memset(f->data[1] + y * f->linesize[1], 128, cw);
    std::memset(f->data[2] + y * f->linesize[2], 128, cw);
  }
}

void ClearP010Frame(AVFrame* f) {
  if (!f || f->format != AV_PIX_FMT_P010LE) return;
  const int w = f->width;
  const int h = f->height;

  const uint16_t y_word = static_cast<uint16_t>(64u << 6);
  const uint16_t uv_word = static_cast<uint16_t>(512u << 6);

  for (int y = 0; y < h; ++y) {
    uint16_t* row = reinterpret_cast<uint16_t*>(f->data[0] + y * f->linesize[0]);
    for (int x = 0; x < w; ++x) row[x] = y_word;
  }

  const int ch = h / 2;
  const int cw = w / 2;
  for (int y = 0; y < ch; ++y) {
    uint16_t* row = reinterpret_cast<uint16_t*>(f->data[1] + y * f->linesize[1]);
    for (int x = 0; x < cw; ++x) {
      row[2 * x + 0] = uv_word;
      row[2 * x + 1] = uv_word;
    }
  }
}

void ClearYuv420p10leFrame(AVFrame* f) {
  if (!f || f->format != AV_PIX_FMT_YUV420P10LE) return;
  const int w = f->width;
  const int h = f->height;

  const uint16_t y_word = 64;
  const uint16_t uv_word = 512;

  for (int y = 0; y < h; ++y) {
    uint16_t* row = reinterpret_cast<uint16_t*>(f->data[0] + y * f->linesize[0]);
    for (int x = 0; x < w; ++x) row[x] = y_word;
  }

  const int cw = w / 2;
  const int ch = h / 2;
  for (int y = 0; y < ch; ++y) {
    uint16_t* urow = reinterpret_cast<uint16_t*>(f->data[1] + y * f->linesize[1]);
    uint16_t* vrow = reinterpret_cast<uint16_t*>(f->data[2] + y * f->linesize[2]);
    for (int x = 0; x < cw; ++x) {
      urow[x] = uv_word;
      vrow[x] = uv_word;
    }
  }
}

void CopyPlaneBytes(const uint8_t* src_data,
                    int src_linesize,
                    uint8_t* dst_data,
                    int dst_linesize,
                    int bytes_per_pixel,
                    const CopyRect& r) {
  if (r.w <= 0 || r.h <= 0) return;
  const int row_bytes = r.w * bytes_per_pixel;
  const int src_xb = r.src_x * bytes_per_pixel;
  const int dst_xb = r.dst_x * bytes_per_pixel;

  for (int y = 0; y < r.h; ++y) {
    const uint8_t* src_line =
        src_data + (r.src_y + y) * src_linesize + src_xb;
    uint8_t* dst_line =
        dst_data + (r.dst_y + y) * dst_linesize + dst_xb;
    std::memcpy(dst_line, src_line, row_bytes);
  }
}

// ---------------- Sub-pixel blit (bilinear) ----------------

inline int ClampInt(int v, int lo, int hi) {
  return std::max(lo, std::min(v, hi));
}

inline double ClampDouble(double v, double lo, double hi) {
  return std::max(lo, std::min(v, hi));
}

// Bilinear blit for 8-bit single-channel plane.
// Places src plane at (dst_x, dst_y) (top-left) in dst plane coordinates.
void BlitPlaneBilinearU8(const uint8_t* src,
                         int src_linesize,
                         int src_w,
                         int src_h,
                         uint8_t* dst,
                         int dst_linesize,
                         int dst_w,
                         int dst_h,
                         double dst_x,
                         double dst_y) {
  if (!src || !dst) return;
  if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) return;

  const int x0 = std::max(0, static_cast<int>(std::ceil(dst_x)));
  const int y0 = std::max(0, static_cast<int>(std::ceil(dst_y)));
  const int x1 = std::min(dst_w - 1, static_cast<int>(std::floor(dst_x + src_w - 1.0)));
  const int y1 = std::min(dst_h - 1, static_cast<int>(std::floor(dst_y + src_h - 1.0)));
  if (x0 > x1 || y0 > y1) return;

  for (int y = y0; y <= y1; ++y) {
    const double sy = static_cast<double>(y) - dst_y;
    const int iy0 = ClampInt(static_cast<int>(std::floor(sy)), 0, src_h - 1);
    const int iy1 = ClampInt(iy0 + 1, 0, src_h - 1);
    const double fy = ClampDouble(sy - std::floor(sy), 0.0, 1.0);

    const uint8_t* row0 = src + iy0 * src_linesize;
    const uint8_t* row1 = src + iy1 * src_linesize;
    uint8_t* drow = dst + y * dst_linesize;

    for (int x = x0; x <= x1; ++x) {
      const double sx = static_cast<double>(x) - dst_x;
      const int ix0 = ClampInt(static_cast<int>(std::floor(sx)), 0, src_w - 1);
      const int ix1 = ClampInt(ix0 + 1, 0, src_w - 1);
      const double fx = ClampDouble(sx - std::floor(sx), 0.0, 1.0);

      const double w00 = (1.0 - fx) * (1.0 - fy);
      const double w10 = fx * (1.0 - fy);
      const double w01 = (1.0 - fx) * fy;
      const double w11 = fx * fy;

      const int p00 = row0[ix0];
      const int p10 = row0[ix1];
      const int p01 = row1[ix0];
      const int p11 = row1[ix1];

      int v = static_cast<int>(std::lround(w00 * p00 + w10 * p10 + w01 * p01 + w11 * p11));
      v = ClampInt(v, 0, 255);
      drow[x] = static_cast<uint8_t>(v);
    }
  }
}

// Bilinear blit for 16-bit single-channel plane (used by YUV420P10LE).
void BlitPlaneBilinearU16(const uint8_t* src_bytes,
                          int src_linesize,
                          int src_w,
                          int src_h,
                          uint8_t* dst_bytes,
                          int dst_linesize,
                          int dst_w,
                          int dst_h,
                          double dst_x,
                          double dst_y) {
  if (!src_bytes || !dst_bytes) return;
  if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) return;

  const int x0 = std::max(0, static_cast<int>(std::ceil(dst_x)));
  const int y0 = std::max(0, static_cast<int>(std::ceil(dst_y)));
  const int x1 = std::min(dst_w - 1, static_cast<int>(std::floor(dst_x + src_w - 1.0)));
  const int y1 = std::min(dst_h - 1, static_cast<int>(std::floor(dst_y + src_h - 1.0)));
  if (x0 > x1 || y0 > y1) return;

  for (int y = y0; y <= y1; ++y) {
    const double sy = static_cast<double>(y) - dst_y;
    const int iy0 = ClampInt(static_cast<int>(std::floor(sy)), 0, src_h - 1);
    const int iy1 = ClampInt(iy0 + 1, 0, src_h - 1);
    const double fy = ClampDouble(sy - std::floor(sy), 0.0, 1.0);

    const uint16_t* row0 =
        reinterpret_cast<const uint16_t*>(src_bytes + iy0 * src_linesize);
    const uint16_t* row1 =
        reinterpret_cast<const uint16_t*>(src_bytes + iy1 * src_linesize);
    uint16_t* drow =
        reinterpret_cast<uint16_t*>(dst_bytes + y * dst_linesize);

    for (int x = x0; x <= x1; ++x) {
      const double sx = static_cast<double>(x) - dst_x;
      const int ix0 = ClampInt(static_cast<int>(std::floor(sx)), 0, src_w - 1);
      const int ix1 = ClampInt(ix0 + 1, 0, src_w - 1);
      const double fx = ClampDouble(sx - std::floor(sx), 0.0, 1.0);

      const double w00 = (1.0 - fx) * (1.0 - fy);
      const double w10 = fx * (1.0 - fy);
      const double w01 = (1.0 - fx) * fy;
      const double w11 = fx * fy;

      const double p00 = static_cast<double>(row0[ix0]);
      const double p10 = static_cast<double>(row0[ix1]);
      const double p01 = static_cast<double>(row1[ix0]);
      const double p11 = static_cast<double>(row1[ix1]);

      double v = w00 * p00 + w10 * p10 + w01 * p01 + w11 * p11;
      int iv = ClampInt(static_cast<int>(std::lround(v)), 0, 65535);
      drow[x] = static_cast<uint16_t>(iv);
    }
  }
}

// Bilinear blit for P010 UV plane (interleaved UV, 16-bit each).
// src_w/src_h and dst_w/dst_h are chroma sizes (width/2, height/2).
void BlitPlaneBilinearP010UV(const uint8_t* src_bytes,
                             int src_linesize,
                             int src_w,
                             int src_h,
                             uint8_t* dst_bytes,
                             int dst_linesize,
                             int dst_w,
                             int dst_h,
                             double dst_x,
                             double dst_y) {
  if (!src_bytes || !dst_bytes) return;
  if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) return;

  const int x0 = std::max(0, static_cast<int>(std::ceil(dst_x)));
  const int y0 = std::max(0, static_cast<int>(std::ceil(dst_y)));
  const int x1 = std::min(dst_w - 1, static_cast<int>(std::floor(dst_x + src_w - 1.0)));
  const int y1 = std::min(dst_h - 1, static_cast<int>(std::floor(dst_y + src_h - 1.0)));
  if (x0 > x1 || y0 > y1) return;

  for (int y = y0; y <= y1; ++y) {
    const double sy = static_cast<double>(y) - dst_y;
    const int iy0 = ClampInt(static_cast<int>(std::floor(sy)), 0, src_h - 1);
    const int iy1 = ClampInt(iy0 + 1, 0, src_h - 1);
    const double fy = ClampDouble(sy - std::floor(sy), 0.0, 1.0);

    const uint16_t* row0 =
        reinterpret_cast<const uint16_t*>(src_bytes + iy0 * src_linesize);
    const uint16_t* row1 =
        reinterpret_cast<const uint16_t*>(src_bytes + iy1 * src_linesize);
    uint16_t* drow =
        reinterpret_cast<uint16_t*>(dst_bytes + y * dst_linesize);

    for (int x = x0; x <= x1; ++x) {
      const double sx = static_cast<double>(x) - dst_x;
      const int ix0 = ClampInt(static_cast<int>(std::floor(sx)), 0, src_w - 1);
      const int ix1 = ClampInt(ix0 + 1, 0, src_w - 1);
      const double fx = ClampDouble(sx - std::floor(sx), 0.0, 1.0);

      const double w00 = (1.0 - fx) * (1.0 - fy);
      const double w10 = fx * (1.0 - fy);
      const double w01 = (1.0 - fx) * fy;
      const double w11 = fx * fy;

      const int o00 = 2 * ix0;
      const int o10 = 2 * ix1;

      const double u00 = static_cast<double>(row0[o00 + 0]);
      const double v00 = static_cast<double>(row0[o00 + 1]);
      const double u10 = static_cast<double>(row0[o10 + 0]);
      const double v10 = static_cast<double>(row0[o10 + 1]);
      const double u01 = static_cast<double>(row1[o00 + 0]);
      const double v01 = static_cast<double>(row1[o00 + 1]);
      const double u11 = static_cast<double>(row1[o10 + 0]);
      const double v11 = static_cast<double>(row1[o10 + 1]);

      const double u = w00 * u00 + w10 * u10 + w01 * u01 + w11 * u11;
      const double v = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11;

      const int du = ClampInt(static_cast<int>(std::lround(u)), 0, 65535);
      const int dv = ClampInt(static_cast<int>(std::lround(v)), 0, 65535);

      drow[2 * x + 0] = static_cast<uint16_t>(du);
      drow[2 * x + 1] = static_cast<uint16_t>(dv);
    }
  }
}


// ---------------- Remux audio (smart: copy AAC, transcode others to AAC) ----------------

static bool VerifyMediaReadable(const std::string& path, std::string* err) {
  AVFormatContext* f = nullptr;
  int ret = avformat_open_input(&f, path.c_str(), nullptr, nullptr);
  if (ret < 0 || !f) {
    if (err) *err = "verify open failed: " + AvErr2Str(ret);
    if (f) avformat_close_input(&f);
    return false;
  }
  ret = avformat_find_stream_info(f, nullptr);
  if (ret < 0) {
    if (err) *err = "verify find_stream_info failed: " + AvErr2Str(ret);
    avformat_close_input(&f);
    return false;
  }
  avformat_close_input(&f);
  return true;
}

static bool IsWindowsFriendlyMp4Audio(AVCodecID id) {
  // Windows 播放器对 MP4+AAC 最稳。PCM(ipcm) 常见不支持。
  return id == AV_CODEC_ID_AAC;
}

static AVSampleFormat PickEncoderSampleFmt(const AVCodec* enc) {
  if (!enc || !enc->sample_fmts) return AV_SAMPLE_FMT_FLTP;
  // Prefer fltp if supported.
  for (const AVSampleFormat* p = enc->sample_fmts; *p != AV_SAMPLE_FMT_NONE; ++p) {
    if (*p == AV_SAMPLE_FMT_FLTP) return *p;
  }
  return enc->sample_fmts[0];
}

static int PickEncoderSampleRate(const AVCodec* enc, int preferred, int fallback) {
  if (!enc || !enc->supported_samplerates) {
    return (preferred > 0) ? preferred : ((fallback > 0) ? fallback : 48000);
  }
  auto Supports = [&](int sr) {
    for (const int* p = enc->supported_samplerates; *p != 0; ++p) {
      if (*p == sr) return true;
    }
    return false;
  };
  if (preferred > 0 && Supports(preferred)) return preferred;
  if (fallback > 0 && Supports(fallback)) return fallback;
  // else pick first
  return enc->supported_samplerates[0] ? enc->supported_samplerates[0] : 48000;
}

static void MakeDefaultLayoutIfUnspec(AVChannelLayout* layout, int nb_channels_fallback) {
  if (!layout) return;
  int ch = layout->nb_channels;
  if (ch <= 0) ch = nb_channels_fallback;
  if (ch <= 0) ch = 2;
  if (layout->nb_channels <= 0) {
    AVChannelLayout def{};
    av_channel_layout_default(&def, ch);
    av_channel_layout_uninit(layout);
    *layout = def;
    return;
  }
  if (layout->order == AV_CHANNEL_ORDER_UNSPEC) {
    AVChannelLayout def{};
    av_channel_layout_default(&def, ch);
    av_channel_layout_uninit(layout);
    *layout = def;
  }
}

static bool RemuxCopyAudioFromSrc(const std::string& src_path,
                                  const std::string& video_only_path,
                                  const std::string& out_path,
                                  std::string* err) {
  AVFormatContext* in_v = nullptr;
  AVFormatContext* in_s = nullptr;
  AVFormatContext* out = nullptr;

  auto Fail = [&](const std::string& e) -> bool {
    if (err) *err = e;
    if (out) {
      if (!(out->oformat->flags & AVFMT_NOFILE) && out->pb) avio_closep(&out->pb);
      avformat_free_context(out);
    }
    if (in_v) avformat_close_input(&in_v);
    if (in_s) avformat_close_input(&in_s);
    return false;
  };

  int ret = avformat_open_input(&in_v, video_only_path.c_str(), nullptr, nullptr);
  if (ret < 0) return Fail("open video_only failed: " + AvErr2Str(ret));
  ret = avformat_find_stream_info(in_v, nullptr);
  if (ret < 0) return Fail("find stream info video_only failed: " + AvErr2Str(ret));

  ret = avformat_open_input(&in_s, src_path.c_str(), nullptr, nullptr);
  if (ret < 0) return Fail("open src failed: " + AvErr2Str(ret));
  ret = avformat_find_stream_info(in_s, nullptr);
  if (ret < 0) return Fail("find stream info src failed: " + AvErr2Str(ret));

  const int v_in_idx = av_find_best_stream(in_v, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  if (v_in_idx < 0) return Fail("no video stream in temp file");

  const int a_in_idx = av_find_best_stream(in_s, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
  if (a_in_idx < 0) return Fail("no audio stream in src");

  AVStream* v_in = in_v->streams[v_in_idx];
  AVStream* a_in = in_s->streams[a_in_idx];

  const AVCodecID a_in_codec = a_in->codecpar->codec_id;
  const bool transcode_audio = !IsWindowsFriendlyMp4Audio(a_in_codec);

  LOG(INFO) << "[VideoExportor] Remux: src audio codec="
            << avcodec_get_name(a_in_codec)
            << " (" << static_cast<int>(a_in_codec) << ")"
            << " transcode_to_aac=" << (transcode_audio ? 1 : 0);

  ret = avformat_alloc_output_context2(&out, nullptr, nullptr, out_path.c_str());
  if (ret < 0 || !out) return Fail("alloc output failed: " + AvErr2Str(ret));

  out->avoid_negative_ts = AVFMT_AVOID_NEG_TS_MAKE_ZERO;

  // ---- output video stream: always stream copy ----
  AVStream* v_out = avformat_new_stream(out, nullptr);
  if (!v_out) return Fail("new video stream failed");
  ret = avcodec_parameters_copy(v_out->codecpar, v_in->codecpar);
  if (ret < 0) return Fail("copy video codecpar failed: " + AvErr2Str(ret));
  v_out->codecpar->codec_tag = 0;
  v_out->time_base = v_in->time_base;

  // ---- output audio stream: copy AAC, otherwise transcode to AAC ----
  AVStream* a_out = avformat_new_stream(out, nullptr);
  if (!a_out) return Fail("new audio stream failed");

  // Muxer opts: faststart (Windows 友好)
  AVDictionary* mux_opt = nullptr;
  av_dict_set(&mux_opt, "movflags", "+faststart", 0);

  if (!(out->oformat->flags & AVFMT_NOFILE)) {
    ret = avio_open(&out->pb, out_path.c_str(), AVIO_FLAG_WRITE);
    if (ret < 0) {
      if (mux_opt) av_dict_free(&mux_opt);
      return Fail("avio_open out failed: " + AvErr2Str(ret));
    }
  }

  // ---------- Common helpers ----------
  const AVRational tb_us{1, AV_TIME_BASE};
  auto TsToUs = [&](const AVPacket* p, AVRational tb) -> int64_t {
    int64_t ts = (p->dts != AV_NOPTS_VALUE) ? p->dts : p->pts;
    if (ts == AV_NOPTS_VALUE) return 0;
    return av_rescale_q(ts, tb, tb_us);
  };

  auto ReadNext = [&](AVFormatContext* f, AVPacket* p, int want_idx) -> bool {
    av_packet_unref(p);
    while (true) {
      int r = av_read_frame(f, p);
      if (r == AVERROR_EOF) return false;
      if (r < 0) return false;
      if (p->stream_index == want_idx) return true;
      av_packet_unref(p);
    }
  };

  auto ShiftAndRescale = [&](AVPacket* p,
                             AVRational in_tb,
                             AVRational out_tb,
                             int out_stream_index,
                             int64_t start_us,
                             int64_t* last_dts_out) {
    const int64_t off = av_rescale_q(start_us, tb_us, in_tb);

    if (p->pts != AV_NOPTS_VALUE) p->pts -= off;
    if (p->dts != AV_NOPTS_VALUE) p->dts -= off;

    if (p->dts != AV_NOPTS_VALUE) {
      int64_t dts_out = av_rescale_q(p->dts, in_tb, out_tb);
      if (*last_dts_out != AV_NOPTS_VALUE && dts_out < *last_dts_out) {
        dts_out = *last_dts_out;
        p->dts = av_rescale_q(dts_out, out_tb, in_tb);
        if (p->pts != AV_NOPTS_VALUE && p->pts < p->dts) {
          p->pts = p->dts;
        }
      }
      *last_dts_out = av_rescale_q(p->dts, in_tb, out_tb);
    }

    p->stream_index = out_stream_index;
    p->pos = -1;
    av_packet_rescale_ts(p, in_tb, out_tb);
  };

  // ---- Prepare audio parameters / encoder if needed ----
  AVCodecContext* a_dec_ctx = nullptr;
  AVCodecContext* a_enc_ctx = nullptr;
  SwrContext* swr = nullptr;
  AVAudioFifo* fifo = nullptr;
  AVFrame* a_dec_frame = nullptr;
  AVFrame* a_res_frame = nullptr;
  AVPacket* a_enc_pkt = nullptr;

  AVSampleFormat out_sample_fmt = AV_SAMPLE_FMT_FLTP;
  int out_sample_rate = 48000;
  AVChannelLayout out_ch_layout{};
  int a_enc_frame_size = 1024;
  int64_t next_a_pts = 0;           // in samples (encoder time_base=1/sr)
  int64_t drop_samples = 0;         // drop initial samples if audio starts before video
  int64_t pad_silence_samples = 0;  // pad initial silence if audio starts after video

  auto CleanupAudio = [&]() {
    if (a_enc_pkt) av_packet_free(&a_enc_pkt);
    if (a_res_frame) av_frame_free(&a_res_frame);
    if (a_dec_frame) av_frame_free(&a_dec_frame);
    if (fifo) av_audio_fifo_free(fifo);
    if (swr) swr_free(&swr);
    if (a_enc_ctx) avcodec_free_context(&a_enc_ctx);
    if (a_dec_ctx) avcodec_free_context(&a_dec_ctx);
    av_channel_layout_uninit(&out_ch_layout);
  };

  auto WriteEncodedAudioPackets = [&]() -> bool {
    if (!a_enc_ctx || !a_enc_pkt) return true;
    while (true) {
      int r = avcodec_receive_packet(a_enc_ctx, a_enc_pkt);
      if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) break;
      if (r < 0) {
        return Fail("audio avcodec_receive_packet failed: " + AvErr2Str(r));
      }
      av_packet_rescale_ts(a_enc_pkt, a_enc_ctx->time_base, a_out->time_base);
      a_enc_pkt->stream_index = a_out->index;
      a_enc_pkt->pos = -1;
      int w = av_interleaved_write_frame(out, a_enc_pkt);
      av_packet_unref(a_enc_pkt);
      if (w < 0) {
        return Fail("write encoded audio pkt failed: " + AvErr2Str(w));
      }
    }
    return true;
  };

  auto EncodeFromFifo = [&](bool flush) -> bool {
    if (!a_enc_ctx || !fifo) return true;

    // AAC 常见要求 frame_size=1024。flush 时把尾巴 pad 到整帧。
    const int frame_size = (a_enc_ctx->frame_size > 0) ? a_enc_ctx->frame_size : a_enc_frame_size;
    if (flush) {
      int sz = av_audio_fifo_size(fifo);
      if (sz > 0 && frame_size > 0) {
        int mod = sz % frame_size;
        if (mod != 0) {
          int need = frame_size - mod;
          // write silence 'need' samples
          AVFrame* sil = av_frame_alloc();
          if (!sil) return Fail("alloc silence frame failed");
          sil->format = a_enc_ctx->sample_fmt;
          sil->sample_rate = a_enc_ctx->sample_rate;
          av_channel_layout_copy(&sil->ch_layout, &a_enc_ctx->ch_layout);
          sil->nb_samples = need;
          int rr = av_frame_get_buffer(sil, 0);
          if (rr < 0) {
            av_frame_free(&sil);
            return Fail("silence av_frame_get_buffer failed: " + AvErr2Str(rr));
          }
          rr = av_frame_make_writable(sil);
          if (rr < 0) {
            av_frame_free(&sil);
            return Fail("silence av_frame_make_writable failed: " + AvErr2Str(rr));
          }
          // memset to 0 -> silence
          for (int i = 0; i < sil->ch_layout.nb_channels; ++i) {
            if (av_sample_fmt_is_planar(a_enc_ctx->sample_fmt)) {
              std::memset(sil->data[i], 0, need * av_get_bytes_per_sample(a_enc_ctx->sample_fmt));
            }
          }
          if (!av_sample_fmt_is_planar(a_enc_ctx->sample_fmt)) {
            std::memset(sil->data[0], 0,
                        need * a_enc_ctx->ch_layout.nb_channels *
                            av_get_bytes_per_sample(a_enc_ctx->sample_fmt));
          }

          // fifo write
          const void* const* src = (const void* const*)sil->data;
          int ww = av_audio_fifo_write(fifo, (void**)src, need);
          av_frame_free(&sil);
          if (ww < need) {
            return Fail("av_audio_fifo_write(silence) failed");
          }
        }
      }
    }

    while (true) {
      const int available = av_audio_fifo_size(fifo);
      if (frame_size > 0) {
        if (available < frame_size) break;
      } else {
        if (available <= 0) break;
      }

      int take = (frame_size > 0) ? frame_size : available;

      AVFrame* enc_frame = av_frame_alloc();
      if (!enc_frame) return Fail("alloc enc_frame failed");
      enc_frame->format = a_enc_ctx->sample_fmt;
      enc_frame->sample_rate = a_enc_ctx->sample_rate;
      av_channel_layout_copy(&enc_frame->ch_layout, &a_enc_ctx->ch_layout);
      enc_frame->nb_samples = take;

      int rr = av_frame_get_buffer(enc_frame, 0);
      if (rr < 0) {
        av_frame_free(&enc_frame);
        return Fail("enc_frame av_frame_get_buffer failed: " + AvErr2Str(rr));
      }

      rr = av_audio_fifo_read(fifo, (void**)enc_frame->data, take);
      if (rr < take) {
        av_frame_free(&enc_frame);
        return Fail("av_audio_fifo_read failed");
      }

      enc_frame->pts = next_a_pts;
      next_a_pts += take;

      rr = avcodec_send_frame(a_enc_ctx, enc_frame);
      av_frame_free(&enc_frame);
      if (rr < 0) {
        return Fail("audio avcodec_send_frame failed: " + AvErr2Str(rr));
      }

      if (!WriteEncodedAudioPackets()) return false;
    }

    if (flush) {
      int rr = avcodec_send_frame(a_enc_ctx, nullptr);
      if (rr < 0 && rr != AVERROR_EOF) {
        return Fail("audio avcodec_send_frame(flush) failed: " + AvErr2Str(rr));
      }
      if (!WriteEncodedAudioPackets()) return false;
    }

    return true;
  };

  auto InitAudioTranscodeToAac = [&]() -> bool {
    // decoder
    const AVCodec* a_dec = avcodec_find_decoder(a_in->codecpar->codec_id);
    if (!a_dec) return Fail("audio decoder not found");

    a_dec_ctx = avcodec_alloc_context3(a_dec);
    if (!a_dec_ctx) return Fail("alloc audio dec ctx failed");
    int r = avcodec_parameters_to_context(a_dec_ctx, a_in->codecpar);
    if (r < 0) return Fail("audio parameters_to_context failed: " + AvErr2Str(r));
    FixupAudioChannelLayoutCtx(a_dec_ctx);

    r = avcodec_open2(a_dec_ctx, a_dec, nullptr);
    if (r < 0) return Fail("open audio decoder failed: " + AvErr2Str(r));

    // encoder AAC
    const AVCodec* a_enc = avcodec_find_encoder(AV_CODEC_ID_AAC);
    if (!a_enc) return Fail("AAC encoder not found");

    a_enc_ctx = avcodec_alloc_context3(a_enc);
    if (!a_enc_ctx) return Fail("alloc audio enc ctx failed");

    // channel layout
    av_channel_layout_copy(&out_ch_layout, &a_dec_ctx->ch_layout);
    MakeDefaultLayoutIfUnspec(&out_ch_layout, a_dec_ctx->ch_layout.nb_channels);

    out_sample_fmt = PickEncoderSampleFmt(a_enc);
    out_sample_rate = PickEncoderSampleRate(a_enc, /*preferred=*/48000, /*fallback=*/a_dec_ctx->sample_rate);

    a_enc_ctx->sample_fmt = out_sample_fmt;
    a_enc_ctx->sample_rate = out_sample_rate;
    av_channel_layout_copy(&a_enc_ctx->ch_layout, &out_ch_layout);
    a_enc_ctx->bit_rate = 192000;
    a_enc_ctx->time_base = AVRational{1, out_sample_rate};

    if (out->oformat->flags & AVFMT_GLOBALHEADER) {
      a_enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    // Some FFmpeg builds want explicit profile, keep default if not needed.
    // a_enc_ctx->profile = FF_PROFILE_AAC_LOW;

    AVDictionary* opt = nullptr;
    r = avcodec_open2(a_enc_ctx, a_enc, &opt);
    if (opt) av_dict_free(&opt);
    if (r < 0) return Fail("open AAC encoder failed: " + AvErr2Str(r));

    a_enc_frame_size = (a_enc_ctx->frame_size > 0) ? a_enc_ctx->frame_size : 1024;

    // output audio stream params from encoder
    r = avcodec_parameters_from_context(a_out->codecpar, a_enc_ctx);
    if (r < 0) return Fail("audio parameters_from_context failed: " + AvErr2Str(r));
    a_out->codecpar->codec_tag = 0;
    a_out->time_base = a_enc_ctx->time_base;

    // swr
    swr = swr_alloc();
    if (!swr) return Fail("swr_alloc failed");

    AVChannelLayout in_layout{};
    av_channel_layout_copy(&in_layout, &a_dec_ctx->ch_layout);
    MakeDefaultLayoutIfUnspec(&in_layout, a_dec_ctx->ch_layout.nb_channels);

    r = swr_alloc_set_opts2(&swr,
                            &a_enc_ctx->ch_layout,
                            a_enc_ctx->sample_fmt,
                            a_enc_ctx->sample_rate,
                            &in_layout,
                            a_dec_ctx->sample_fmt,
                            a_dec_ctx->sample_rate,
                            0, nullptr);
    av_channel_layout_uninit(&in_layout);
    if (r < 0) return Fail("swr_alloc_set_opts2 failed: " + AvErr2Str(r));

    r = swr_init(swr);
    if (r < 0) return Fail("swr_init failed: " + AvErr2Str(r));

    // fifo
    fifo = av_audio_fifo_alloc(a_enc_ctx->sample_fmt,
                               a_enc_ctx->ch_layout.nb_channels,
                               a_enc_frame_size * 8);
    if (!fifo) return Fail("av_audio_fifo_alloc failed");

    a_dec_frame = av_frame_alloc();
    a_res_frame = av_frame_alloc();
    a_enc_pkt = av_packet_alloc();
    if (!a_dec_frame || !a_res_frame || !a_enc_pkt) return Fail("alloc audio frame/pkt failed");

    LOG(INFO) << "[VideoExportor] Audio transcode: "
              << "dec=" << (a_dec->name ? a_dec->name : "unknown")
              << " -> enc=aac"
              << " sr=" << a_enc_ctx->sample_rate
              << " fmt=" << av_get_sample_fmt_name(a_enc_ctx->sample_fmt)
              << " ch=" << a_enc_ctx->ch_layout.nb_channels
              << " frame_size=" << a_enc_frame_size;

    return true;
  };

  auto FeedAudioPacket = [&](AVPacket* in_pkt, int64_t start_us, int64_t audio_first_us) -> bool {
    if (!a_dec_ctx || !swr || !fifo) return Fail("audio transcode not initialized");

    // Compute pad/drop once based on first audio packet time
    static bool offset_inited = false;
    if (!offset_inited) {
      const int64_t offset_us = audio_first_us - start_us;  // audio relative to shared start
      const int64_t offset_samples = (offset_us * (int64_t)out_sample_rate) / (int64_t)AV_TIME_BASE;
      if (offset_samples > 0) {
        pad_silence_samples = offset_samples;
        drop_samples = 0;
      } else {
        pad_silence_samples = 0;
        drop_samples = -offset_samples;
      }
      offset_inited = true;

      if (pad_silence_samples > 0) {
        // push silence into fifo to delay audio
        int remain = (int)pad_silence_samples;
        while (remain > 0) {
          int chunk = std::min(remain, a_enc_frame_size);
          AVFrame* sil = av_frame_alloc();
          if (!sil) return Fail("alloc silence frame failed");
          sil->format = a_enc_ctx->sample_fmt;
          sil->sample_rate = a_enc_ctx->sample_rate;
          av_channel_layout_copy(&sil->ch_layout, &a_enc_ctx->ch_layout);
          sil->nb_samples = chunk;
          int rr = av_frame_get_buffer(sil, 0);
          if (rr < 0) {
            av_frame_free(&sil);
            return Fail("silence get_buffer failed: " + AvErr2Str(rr));
          }
          rr = av_frame_make_writable(sil);
          if (rr < 0) {
            av_frame_free(&sil);
            return Fail("silence make_writable failed: " + AvErr2Str(rr));
          }
          // memset 0
          const int bps = av_get_bytes_per_sample(a_enc_ctx->sample_fmt);
          if (av_sample_fmt_is_planar(a_enc_ctx->sample_fmt)) {
            for (int c = 0; c < sil->ch_layout.nb_channels; ++c) {
              std::memset(sil->data[c], 0, chunk * bps);
            }
          } else {
            std::memset(sil->data[0], 0, chunk * bps * sil->ch_layout.nb_channels);
          }

          const void* const* src = (const void* const*)sil->data;
          int ww = av_audio_fifo_write(fifo, (void**)src, chunk);
          av_frame_free(&sil);
          if (ww < chunk) return Fail("fifo write silence failed");
          remain -= chunk;
        }
        pad_silence_samples = 0;
      }

      LOG(INFO) << "[VideoExportor] A/V offset (us)=" << (audio_first_us - start_us)
                << " => drop_samples=" << drop_samples;
    }

    int rr = avcodec_send_packet(a_dec_ctx, in_pkt);
    if (rr < 0) return Fail("audio avcodec_send_packet failed: " + AvErr2Str(rr));

    while (true) {
      rr = avcodec_receive_frame(a_dec_ctx, a_dec_frame);
      if (rr == AVERROR(EAGAIN) || rr == AVERROR_EOF) break;
      if (rr < 0) return Fail("audio avcodec_receive_frame failed: " + AvErr2Str(rr));

      FixupAudioChannelLayoutCtx(a_dec_ctx);

      // resample
      av_frame_unref(a_res_frame);
      a_res_frame->format = a_enc_ctx->sample_fmt;
      a_res_frame->sample_rate = a_enc_ctx->sample_rate;
      av_channel_layout_copy(&a_res_frame->ch_layout, &a_enc_ctx->ch_layout);

      int64_t delay = swr_get_delay(swr, a_dec_ctx->sample_rate);
      int out_nb = (int)av_rescale_rnd(delay + a_dec_frame->nb_samples,
                                       a_enc_ctx->sample_rate,
                                       a_dec_ctx->sample_rate,
                                       AV_ROUND_UP);
      a_res_frame->nb_samples = out_nb;

      rr = av_frame_get_buffer(a_res_frame, 0);
      if (rr < 0) return Fail("resampled av_frame_get_buffer failed: " + AvErr2Str(rr));

      rr = swr_convert(swr,
                       a_res_frame->data, a_res_frame->nb_samples,
                       (const uint8_t**)a_dec_frame->data, a_dec_frame->nb_samples);
      if (rr < 0) return Fail("swr_convert failed: " + AvErr2Str(rr));
      a_res_frame->nb_samples = rr;

      // drop leading samples if audio starts earlier than video
      int write_samples = a_res_frame->nb_samples;
      int skip = 0;
      if (drop_samples > 0) {
        skip = (int)std::min<int64_t>(drop_samples, write_samples);
        drop_samples -= skip;
        write_samples -= skip;
      }

      if (write_samples > 0) {
        const int bps = av_get_bytes_per_sample(a_enc_ctx->sample_fmt);
        uint8_t* tmp_data[AV_NUM_DATA_POINTERS] = {0};

        if (av_sample_fmt_is_planar(a_enc_ctx->sample_fmt)) {
          for (int c = 0; c < a_enc_ctx->ch_layout.nb_channels; ++c) {
            tmp_data[c] = a_res_frame->data[c] + skip * bps;
          }
        } else {
          tmp_data[0] = a_res_frame->data[0] + skip * bps * a_enc_ctx->ch_layout.nb_channels;
        }

        const void* const* src = (const void* const*)tmp_data;
        int ww = av_audio_fifo_write(fifo, (void**)src, write_samples);
        if (ww < write_samples) return Fail("av_audio_fifo_write failed");
      }

      av_frame_unref(a_dec_frame);

      // encode as available
      if (!EncodeFromFifo(/*flush=*/false)) return false;
    }

    return true;
  };

  // ---------- Peek first packets to compute shared start_us ----------
  AVPacket* pv = av_packet_alloc();
  AVPacket* pa = av_packet_alloc();
  if (!pv || !pa) {
    if (pv) av_packet_free(&pv);
    if (pa) av_packet_free(&pa);
    if (mux_opt) av_dict_free(&mux_opt);
    return Fail("alloc packets failed");
  }

  bool has_v = ReadNext(in_v, pv, v_in_idx);
  bool has_a = ReadNext(in_s, pa, a_in_idx);

  int64_t start_us = 0;
  int64_t first_v_us = 0;
  int64_t first_a_us = 0;
  if (has_v) first_v_us = TsToUs(pv, v_in->time_base);
  if (has_a) first_a_us = TsToUs(pa, a_in->time_base);

  if (has_v && has_a) start_us = std::min(first_v_us, first_a_us);
  else if (has_v) start_us = first_v_us;
  else if (has_a) start_us = first_a_us;

  // ---- Setup output audio stream + write header (depends on transcode or copy) ----
  if (!transcode_audio) {
    // stream copy audio
    ret = avcodec_parameters_copy(a_out->codecpar, a_in->codecpar);
    if (ret < 0) {
      av_packet_free(&pv);
      av_packet_free(&pa);
      if (mux_opt) av_dict_free(&mux_opt);
      return Fail("copy audio codecpar failed: " + AvErr2Str(ret));
    }
    a_out->codecpar->codec_tag = 0;
    FixupAudioChannelLayout(a_out->codecpar);
    a_out->time_base = a_in->time_base;

    LOG(INFO) << "[VideoExportor] Audio stream-copy: codec_id=" << a_out->codecpar->codec_id
              << " codec=" << avcodec_get_name(a_out->codecpar->codec_id)
              << " ch=" << a_out->codecpar->ch_layout.nb_channels
              << " order=" << static_cast<int>(a_out->codecpar->ch_layout.order);
  } else {
    // transcode to AAC
    if (!InitAudioTranscodeToAac()) {
      av_packet_free(&pv);
      av_packet_free(&pa);
      if (mux_opt) av_dict_free(&mux_opt);
      CleanupAudio();
      return false;
    }
  }

  ret = avformat_write_header(out, mux_opt ? &mux_opt : nullptr);
  if (mux_opt) av_dict_free(&mux_opt);
  if (ret < 0) {
    av_packet_free(&pv);
    av_packet_free(&pa);
    CleanupAudio();
    return Fail("write header failed: " + AvErr2Str(ret));
  }

  // DTS monotonic trackers
  int64_t last_v_dts_out = AV_NOPTS_VALUE;
  int64_t last_a_dts_out = AV_NOPTS_VALUE;

  // ---------- Main interleave loop ----------
  while (has_v || has_a) {
    bool take_v = false;
    if (has_v && has_a) {
      take_v = (TsToUs(pv, v_in->time_base) <= TsToUs(pa, a_in->time_base));
    } else {
      take_v = has_v;
    }

    if (take_v) {
      ShiftAndRescale(pv, v_in->time_base, v_out->time_base, v_out->index, start_us, &last_v_dts_out);
      int w = av_interleaved_write_frame(out, pv);
      if (w < 0) {
        av_packet_free(&pv);
        av_packet_free(&pa);
        CleanupAudio();
        return Fail("write video pkt failed: " + AvErr2Str(w));
      }
      has_v = ReadNext(in_v, pv, v_in_idx);
    } else {
      if (!transcode_audio) {
        ShiftAndRescale(pa, a_in->time_base, a_out->time_base, a_out->index, start_us, &last_a_dts_out);
        int w = av_interleaved_write_frame(out, pa);
        if (w < 0) {
          av_packet_free(&pv);
          av_packet_free(&pa);
          CleanupAudio();
          return Fail("write audio pkt failed: " + AvErr2Str(w));
        }
      } else {
        // feed packet to decoder -> resample -> fifo -> encode -> write
        const int64_t audio_pkt_us = TsToUs(pa, a_in->time_base);
        if (!FeedAudioPacket(pa, start_us, first_a_us)) {
          av_packet_free(&pv);
          av_packet_free(&pa);
          CleanupAudio();
          return false;
        }
        // Keep last_a_dts_out unused in transcode path.
      }
      has_a = ReadNext(in_s, pa, a_in_idx);
    }
  }

  // ---------- Flush audio transcode pipeline ----------
  if (transcode_audio) {
    // flush decoder
    int rr = avcodec_send_packet(a_dec_ctx, nullptr);
    if (rr < 0 && rr != AVERROR_EOF) {
      av_packet_free(&pv);
      av_packet_free(&pa);
      CleanupAudio();
      return Fail("audio send_packet(flush) failed: " + AvErr2Str(rr));
    }
    while (true) {
      rr = avcodec_receive_frame(a_dec_ctx, a_dec_frame);
      if (rr == AVERROR_EOF || rr == AVERROR(EAGAIN)) break;
      if (rr < 0) {
        av_packet_free(&pv);
        av_packet_free(&pa);
        CleanupAudio();
        return Fail("audio receive_frame(flush) failed: " + AvErr2Str(rr));
      }

      // resample
      av_frame_unref(a_res_frame);
      a_res_frame->format = a_enc_ctx->sample_fmt;
      a_res_frame->sample_rate = a_enc_ctx->sample_rate;
      av_channel_layout_copy(&a_res_frame->ch_layout, &a_enc_ctx->ch_layout);

      int64_t delay = swr_get_delay(swr, a_dec_ctx->sample_rate);
      int out_nb = (int)av_rescale_rnd(delay + a_dec_frame->nb_samples,
                                       a_enc_ctx->sample_rate,
                                       a_dec_ctx->sample_rate,
                                       AV_ROUND_UP);
      a_res_frame->nb_samples = out_nb;

      int r2 = av_frame_get_buffer(a_res_frame, 0);
      if (r2 < 0) {
        av_packet_free(&pv);
        av_packet_free(&pa);
        CleanupAudio();
        return Fail("resampled get_buffer(flush) failed: " + AvErr2Str(r2));
      }

      r2 = swr_convert(swr,
                       a_res_frame->data, a_res_frame->nb_samples,
                       (const uint8_t**)a_dec_frame->data, a_dec_frame->nb_samples);
      if (r2 < 0) {
        av_packet_free(&pv);
        av_packet_free(&pa);
        CleanupAudio();
        return Fail("swr_convert(flush) failed: " + AvErr2Str(r2));
      }
      a_res_frame->nb_samples = r2;

      int write_samples = a_res_frame->nb_samples;
      int skip = 0;
      if (drop_samples > 0) {
        skip = (int)std::min<int64_t>(drop_samples, write_samples);
        drop_samples -= skip;
        write_samples -= skip;
      }

      if (write_samples > 0) {
        const int bps = av_get_bytes_per_sample(a_enc_ctx->sample_fmt);
        uint8_t* tmp_data[AV_NUM_DATA_POINTERS] = {0};

        if (av_sample_fmt_is_planar(a_enc_ctx->sample_fmt)) {
          for (int c = 0; c < a_enc_ctx->ch_layout.nb_channels; ++c) {
            tmp_data[c] = a_res_frame->data[c] + skip * bps;
          }
        } else {
          tmp_data[0] = a_res_frame->data[0] + skip * bps * a_enc_ctx->ch_layout.nb_channels;
        }

        const void* const* src = (const void* const*)tmp_data;
        int ww = av_audio_fifo_write(fifo, (void**)src, write_samples);
        if (ww < write_samples) {
          av_packet_free(&pv);
          av_packet_free(&pa);
          CleanupAudio();
          return Fail("fifo write(flush) failed");
        }
      }

      av_frame_unref(a_dec_frame);
      if (!EncodeFromFifo(/*flush=*/false)) {
        av_packet_free(&pv);
        av_packet_free(&pa);
        CleanupAudio();
        return false;
      }
    }

    // flush encoder + drain fifo
    if (!EncodeFromFifo(/*flush=*/true)) {
      av_packet_free(&pv);
      av_packet_free(&pa);
      CleanupAudio();
      return false;
    }
  }

  // Trailer must be checked. This was your critical bug (you already fixed it).
  int tret = av_write_trailer(out);
  if (tret < 0) {
    av_packet_free(&pv);
    av_packet_free(&pa);
    CleanupAudio();
    return Fail("write trailer failed: " + AvErr2Str(tret));
  }

  av_packet_free(&pv);
  av_packet_free(&pa);

  if (!(out->oformat->flags & AVFMT_NOFILE) && out->pb) {
    int c = avio_closep(&out->pb);
    if (c < 0) {
      CleanupAudio();
      return Fail("avio_closep(out) failed: " + AvErr2Str(c));
    }
  }

  avformat_free_context(out);
  out = nullptr;

  avformat_close_input(&in_v);
  avformat_close_input(&in_s);

  CleanupAudio();

  std::string vfy_err;
  if (!VerifyMediaReadable(out_path, &vfy_err)) {
    return Fail("final output not readable: " + vfy_err);
  }

  return true;
}

}  // namespace

// ---------------- VideoExportor ----------------

VideoExportor::VideoExportor() = default;

VideoExportor::~VideoExportor() {
  Stop();
}

void VideoExportor::Stop() {
  stop_ = true;
  if (worker_.joinable()) worker_.join();
  running_ = false;
}

void VideoExportor::AddExportCallback(ExportProgressCallback cb) {
  std::lock_guard<std::mutex> lk(cb_mtx_);
  progress_cbs_.push_back(std::move(cb));
}

void VideoExportor::AddExportDoneCallback(std::function<void()> cb) {
  std::lock_guard<std::mutex> lk(cb_mtx_);
  done_cbs_.push_back(std::move(cb));
}

void VideoExportor::StartExport(const std::string& src_video_path,
                                const VideoInfo& video_info,
                                const ExportParams& export_param,
                                const std::vector<FrameStableResult>& stable_results) {
  Stop();

  stop_ = false;
  running_ = true;

  worker_ = std::thread(&VideoExportor::WorkerMain,
                        this,
                        src_video_path,
                        video_info,
                        export_param,
                        stable_results);
}

void VideoExportor::WorkerMain(std::string src_video_path,
                               VideoInfo video_info,
                               ExportParams export_param,
                               std::vector<FrameStableResult> stable_results) {
  LOG(INFO) << "[VideoExportor] Worker start. src=" << src_video_path
            << " dst=" << export_param.export_path;

  const std::string final_out = export_param.export_path;
  const std::string tmp_video_only = export_param.export_path + ".video_only.mp4";

  ExportDecoder dec;
  std::string err;
  VideoInfo src_info = video_info;
  if (!dec.Open(src_video_path, &src_info, &err)) {
    LOG(ERROR) << "[VideoExportor] Decoder open failed: " << err;
    running_ = false;
    return;
  }

  const int src_w = dec.src_width();
  const int src_h = dec.src_height();

  int nframes = 0;
  std::vector<double> shift_x;
  std::vector<double> shift_y;
  CanvasPlan plan = ComputeCanvasPlanAndShifts(
      src_info, src_w, src_h, stable_results,
      &nframes, &shift_x, &shift_y);

  LOG(INFO) << "[VideoExportor] CanvasPlan canvas="
            << plan.canvas_w << "x" << plan.canvas_h
            << " origin=(" << plan.origin_x << "," << plan.origin_y << ")"
            << " nframes=" << nframes;

  int out_w = plan.canvas_w;
  int out_h = plan.canvas_h;

  int64_t out_bitrate = 0;
  if (export_param.export_bitrate > 0.0) {
    out_bitrate = static_cast<int64_t>(export_param.export_bitrate);
  } else if (src_info.bitrate > 0.0) {
    out_bitrate = static_cast<int64_t>(src_info.bitrate);
  } else {
    out_bitrate = 8 * 1000 * 1000;
  }

  // --- Perf stats ---
  struct PerfAcc {
    int frames = 0;
    double ms_decode = 0;
    double ms_sws = 0;
    double ms_clear = 0;
    double ms_copy = 0;
    double ms_encode = 0;
    double ms_cb = 0;
    double ms_total = 0;

    Clock::time_point t_start = Clock::now();

    void Add(double d, double s, double c, double cp, double e, double cb, double total) {
      ++frames;
      ms_decode += d;
      ms_sws += s;
      ms_clear += c;
      ms_copy += cp;
      ms_encode += e;
      ms_cb += cb;
      ms_total += total;
    }

    void MaybePrint(const std::string& encoder,
                    AVPixelFormat dec_fmt,
                    AVPixelFormat out_fmt,
                    int out_w,
                    int out_h) {
      constexpr int kPrintEveryN = 60;
      if (frames % kPrintEveryN != 0) return;

      const auto now = Clock::now();
      const double sec = std::chrono::duration<double>(now - t_start).count();
      const double fps = (sec > 1e-9) ? (frames / sec) : 0.0;

      const double denom = std::max(frames, 1);
      const double avg_decode = ms_decode / denom;
      const double avg_sws = ms_sws / denom;
      const double avg_clear = ms_clear / denom;
      const double avg_copy = ms_copy / denom;
      const double avg_encode = ms_encode / denom;
      const double avg_cb = ms_cb / denom;
      const double avg_total = ms_total / denom;

      const int64_t frame_bytes = FrameBytesForPixFmt(out_w, out_h, out_fmt);
      const double bytes_per_frame_est = static_cast<double>(frame_bytes) * 5.0;
      const double bw_gbs = (avg_total > 1e-9)
                                ? (bytes_per_frame_est / (avg_total / 1000.0)) / 1e9
                                : 0.0;

      LOG(INFO) << "[VideoExportor][Perf] frames=" << frames
                << " fps=" << fps
                << " encoder=" << encoder
                << " out=" << out_w << "x" << out_h
                << " dec_fmt=" << av_get_pix_fmt_name(dec_fmt)
                << " out_fmt=" << av_get_pix_fmt_name(out_fmt)
                << " avg_ms(decode/sws/clear/copy/encode/cb/total)="
                << avg_decode << "/" << avg_sws << "/" << avg_clear << "/"
                << avg_copy << "/" << avg_encode << "/" << avg_cb << "/"
                << avg_total
                << " mem_bw_est~" << bw_gbs << " GB/s";
    }
  } perf;

  // ---- Decode first frame to decide 8/10-bit pipeline ----
  AVFrame* first_dec_frame = nullptr;
  if (!dec.DecodeOneFrame(&first_dec_frame) || !first_dec_frame) {
    LOG(ERROR) << "[VideoExportor] Decode failed at first frame (empty video?)";
    running_ = false;
    return;
  }

  const AVPixelFormat first_fmt = static_cast<AVPixelFormat>(first_dec_frame->format);
  const int first_depth = GetPixFmtBitDepth(first_fmt);
  const bool input_10bit = (first_depth > 8);

  LOG(INFO) << "[VideoExportor] First frame fmt=" << av_get_pix_fmt_name(first_fmt)
            << " depth=" << first_depth
            << " size=" << first_dec_frame->width << "x" << first_dec_frame->height
            << " input_10bit=" << (input_10bit ? 1 : 0);

  AVPixelFormat preferred_out_fmt = input_10bit ? AV_PIX_FMT_P010LE : AV_PIX_FMT_YUV420P;

  std::vector<const char*> enc_names;
  if (!input_10bit) {
    enc_names = {"h264_nvenc", "h264_qsv", "h264_amf", "h264_mf", "libx264", "h264", "mpeg4"};
  } else {
    enc_names = {"hevc_nvenc", "hevc_qsv", "hevc_amf", "hevc_mf", "libx265", "hevc"};
  }

  ExportVideoWriter writer;
  try {
    writer.Open(tmp_video_only,
                out_w,
                out_h,
                dec.fps_q(),          // FIX #2: exact rational fps
                out_bitrate,
                preferred_out_fmt,
                enc_names);
  } catch (const std::exception& e) {
    LOG(ERROR) << "[VideoExportor] Writer open failed: " << e.what();
    av_frame_unref(first_dec_frame);
    running_ = false;
    return;
  }

  const AVPixelFormat out_fmt = writer.pix_fmt();
  const int out_depth = GetPixFmtBitDepth(out_fmt);

  LOG(INFO) << "[VideoExportor] Export output pix_fmt=" << av_get_pix_fmt_name(out_fmt)
            << " depth=" << out_depth
            << " encoder=" << writer.encoder_name();

  if (input_10bit && out_depth <= 8) {
    LOG(ERROR) << "[VideoExportor] Refusing to proceed: input is 10-bit but selected output is <=8-bit.";
    writer.Close();
    av_frame_unref(first_dec_frame);
    running_ = false;
    return;
  }

  // ---- Optional sws conversion (only if needed) ----
  SwsContext* sws_to_out = nullptr;
  AVFrame* src_conv = nullptr;

  auto EnsureSrcConvAllocated = [&](int w, int h) -> bool {
    if (src_conv) return true;
    src_conv = av_frame_alloc();
    if (!src_conv) return false;
    src_conv->format = out_fmt;
    src_conv->width = w;
    src_conv->height = h;
    int r = av_frame_get_buffer(src_conv, 32);
    if (r < 0) {
      LOG(ERROR) << "[VideoExportor] av_frame_get_buffer(src_conv) failed: " << AvErr2Str(r);
      return false;
    }
    return true;
  };

  auto PrepareSwsIfNeeded = [&](const AVFrame* in_frame) -> bool {
    if (!in_frame) return false;
    const AVPixelFormat in_fmt = static_cast<AVPixelFormat>(in_frame->format);

    if (in_frame->width == src_w && in_frame->height == src_h && in_fmt == out_fmt) {
      return true;
    }

    if (!sws_to_out) {
      sws_to_out = sws_getContext(
          in_frame->width,
          in_frame->height,
          in_fmt,
          src_w,
          src_h,
          out_fmt,
          SWS_FAST_BILINEAR,
          nullptr,
          nullptr,
          nullptr);
      if (!sws_to_out) {
        LOG(ERROR) << "[VideoExportor] sws_getContext failed. in_fmt="
                   << av_get_pix_fmt_name(in_fmt)
                   << " out_fmt=" << av_get_pix_fmt_name(out_fmt);
        return false;
      }
      LOG(INFO) << "[VideoExportor] sws_to_out created: in_fmt="
                << av_get_pix_fmt_name(in_fmt)
                << " in_size=" << in_frame->width << "x" << in_frame->height
                << " out_fmt=" << av_get_pix_fmt_name(out_fmt)
                << " out_size=" << src_w << "x" << src_h;
    }
    return true;
  };

  auto ConvertIfNeeded = [&](AVFrame* in_frame, double* out_ms_sws) -> AVFrame* {
    *out_ms_sws = 0.0;
    if (!in_frame) return nullptr;
    const AVPixelFormat in_fmt = static_cast<AVPixelFormat>(in_frame->format);

    if (in_frame->width == src_w && in_frame->height == src_h && in_fmt == out_fmt) {
      return in_frame;
    }

    if (!PrepareSwsIfNeeded(in_frame)) return nullptr;
    if (!EnsureSrcConvAllocated(src_w, src_h)) return nullptr;

    int r = av_frame_make_writable(src_conv);
    if (r < 0) {
      LOG(ERROR) << "[VideoExportor] av_frame_make_writable(src_conv) failed: " << AvErr2Str(r);
      return nullptr;
    }

    const uint8_t* src_data[4] = {in_frame->data[0], in_frame->data[1], in_frame->data[2], in_frame->data[3]};
    int src_linesize[4] = {in_frame->linesize[0], in_frame->linesize[1], in_frame->linesize[2], in_frame->linesize[3]};

    uint8_t* dst_data[4] = {src_conv->data[0], src_conv->data[1], src_conv->data[2], src_conv->data[3]};
    int dst_linesize[4] = {src_conv->linesize[0], src_conv->linesize[1], src_conv->linesize[2], src_conv->linesize[3]};

    auto t0 = Clock::now();
    int sret = sws_scale(sws_to_out,
                         src_data,
                         src_linesize,
                         0,
                         in_frame->height,
                         dst_data,
                         dst_linesize);
    auto t1 = Clock::now();
    *out_ms_sws = MsSince(t0, t1);

    if (sret <= 0) {
      LOG(ERROR) << "[VideoExportor] sws_scale failed, ret=" << sret
                 << " in_fmt=" << av_get_pix_fmt_name(in_fmt)
                 << " out_fmt=" << av_get_pix_fmt_name(out_fmt);
      return nullptr;
    }

    return src_conv;
  };

  // ---- Allocate canvas in out_fmt ----
  AVFrame* canvas = av_frame_alloc();
  if (!canvas) {
    LOG(ERROR) << "[VideoExportor] av_frame_alloc(canvas) failed";
    if (src_conv) av_frame_free(&src_conv);
    writer.Close();
    av_frame_unref(first_dec_frame);
    running_ = false;
    return;
  }
  canvas->format = out_fmt;
  canvas->width = out_w;
  canvas->height = out_h;
  int ret = av_frame_get_buffer(canvas, 32);
  if (ret < 0) {
    LOG(ERROR) << "[VideoExportor] av_frame_get_buffer(canvas) failed: " << AvErr2Str(ret);
    av_frame_free(&canvas);
    if (src_conv) av_frame_free(&src_conv);
    writer.Close();
    av_frame_unref(first_dec_frame);
    running_ = false;
    return;
  }

  auto ClearCanvas = [&](AVFrame* f) {
    if (!f) return;
    if (f->format == AV_PIX_FMT_YUV420P) ClearYuv420pFrame(f);
    else if (f->format == AV_PIX_FMT_P010LE) ClearP010Frame(f);
    else if (f->format == AV_PIX_FMT_YUV420P10LE) ClearYuv420p10leFrame(f);
    else LOG(ERROR) << "[VideoExportor] ClearCanvas unsupported fmt=" << av_get_pix_fmt_name((AVPixelFormat)f->format);
  };

    auto CopyToCanvas = [&](const AVFrame* src, AVFrame* dst, double dst_x_y, double dst_y_y) {
    if (!src || !dst) return;

    if (dst->format == AV_PIX_FMT_YUV420P) {
      BlitPlaneBilinearU8(src->data[0], src->linesize[0], src_w, src_h,
                          dst->data[0], dst->linesize[0], out_w, out_h,
                          dst_x_y, dst_y_y);

      const int src_w_c = src_w / 2;
      const int src_h_c = src_h / 2;
      const int out_w_c = out_w / 2;
      const int out_h_c = out_h / 2;
      const double dst_x_c = dst_x_y * 0.5;
      const double dst_y_c = dst_y_y * 0.5;

      BlitPlaneBilinearU8(src->data[1], src->linesize[1], src_w_c, src_h_c,
                          dst->data[1], dst->linesize[1], out_w_c, out_h_c,
                          dst_x_c, dst_y_c);
      BlitPlaneBilinearU8(src->data[2], src->linesize[2], src_w_c, src_h_c,
                          dst->data[2], dst->linesize[2], out_w_c, out_h_c,
                          dst_x_c, dst_y_c);
      return;
    }

    if (dst->format == AV_PIX_FMT_P010LE) {
      BlitPlaneBilinearU16(src->data[0], src->linesize[0], src_w, src_h,
                           dst->data[0], dst->linesize[0], out_w, out_h,
                           dst_x_y, dst_y_y);

      const int src_w_c = src_w / 2;
      const int src_h_c = src_h / 2;
      const int out_w_c = out_w / 2;
      const int out_h_c = out_h / 2;
      const double dst_x_c = dst_x_y * 0.5;
      const double dst_y_c = dst_y_y * 0.5;

      BlitPlaneBilinearP010UV(src->data[1], src->linesize[1], src_w_c, src_h_c,
                              dst->data[1], dst->linesize[1], out_w_c, out_h_c,
                              dst_x_c, dst_y_c);
      return;
    }

    if (dst->format == AV_PIX_FMT_YUV420P10LE) {
      BlitPlaneBilinearU16(src->data[0], src->linesize[0], src_w, src_h,
                           dst->data[0], dst->linesize[0], out_w, out_h,
                           dst_x_y, dst_y_y);

      const int src_w_c = src_w / 2;
      const int src_h_c = src_h / 2;
      const int out_w_c = out_w / 2;
      const int out_h_c = out_h / 2;
      const double dst_x_c = dst_x_y * 0.5;
      const double dst_y_c = dst_y_y * 0.5;

      BlitPlaneBilinearU16(src->data[1], src->linesize[1], src_w_c, src_h_c,
                           dst->data[1], dst->linesize[1], out_w_c, out_h_c,
                           dst_x_c, dst_y_c);
      BlitPlaneBilinearU16(src->data[2], src->linesize[2], src_w_c, src_h_c,
                           dst->data[2], dst->linesize[2], out_w_c, out_h_c,
                           dst_x_c, dst_y_c);
      return;
    }

    LOG(ERROR) << "[VideoExportor] CopyToCanvas unsupported dst fmt="
               << av_get_pix_fmt_name(static_cast<AVPixelFormat>(dst->format));
  };


  int frame_idx = 0;

  // First frame already decoded; process it first.
  AVFrame* pending = first_dec_frame;

  while (!stop_) {
    const auto t_total0 = Clock::now();

    double ms_decode = 0.0;
    double ms_sws = 0.0;
    double ms_clear = 0.0;
    double ms_copy = 0.0;
    double ms_encode = 0.0;
    double ms_cb = 0.0;

    // ---- Decode ----
    AVFrame* dec_frame = nullptr;
    if (pending) {
      dec_frame = pending;
      pending = nullptr;
    } else {
      auto t0 = Clock::now();
      if (!dec.DecodeOneFrame(&dec_frame)) {
        LOG(INFO) << "[VideoExportor] Decode EOF.";
        break;
      }
      auto t1 = Clock::now();
      ms_decode = MsSince(t0, t1);
    }

    if (!dec_frame) {
      LOG(ERROR) << "[VideoExportor] Decode returned null frame.";
      break;
    }

    const AVPixelFormat dec_fmt = static_cast<AVPixelFormat>(dec_frame->format);

    // ---- Convert (only if needed) ----
    AVFrame* src_frame = ConvertIfNeeded(dec_frame, &ms_sws);
    if (!src_frame) {
      LOG(ERROR) << "[VideoExportor] ConvertIfNeeded failed at frame=" << frame_idx;
      break;
    }

    // ---- Clear canvas ----
    {
      auto t0 = Clock::now();
      ret = av_frame_make_writable(canvas);
      if (ret < 0) {
        LOG(ERROR) << "[VideoExportor] av_frame_make_writable(canvas) failed: " << AvErr2Str(ret);
        break;
      }
      ClearCanvas(canvas);
      auto t1 = Clock::now();
      ms_clear = MsSince(t0, t1);
    }

    // ---- Copy into canvas ----
    double sx = 0.0, sy = 0.0;
    if (frame_idx >= 0 && frame_idx < static_cast<int>(shift_x.size())) {
      sx = shift_x[frame_idx];
      sy = shift_y[frame_idx];
    }
    const double dst_x_y = static_cast<double>(plan.origin_x) + sx;
    const double dst_y_y = static_cast<double>(plan.origin_y) + sy;

    {
      auto t0 = Clock::now();
      CopyToCanvas(src_frame, canvas, dst_x_y, dst_y_y);
      auto t1 = Clock::now();
      ms_copy = MsSince(t0, t1);
    }

    // ---- Encode ----
    {
      auto t0 = Clock::now();
      canvas->pts = frame_idx;
      writer.WriteFrame(canvas);
      auto t1 = Clock::now();
      ms_encode = MsSince(t0, t1);
    }

    // ---- Callback ----
    {
      auto t0 = Clock::now();
      std::lock_guard<std::mutex> lk(cb_mtx_);
      for (const auto& cb : progress_cbs_) {
        if (cb) cb(frame_idx);
      }
      auto t1 = Clock::now();
      ms_cb = MsSince(t0, t1);
    }

    const auto t_total1 = Clock::now();
    const double ms_total = MsSince(t_total0, t_total1);

    perf.Add(ms_decode, ms_sws, ms_clear, ms_copy, ms_encode, ms_cb, ms_total);
    perf.MaybePrint(writer.encoder_name(), dec_fmt, out_fmt, out_w, out_h);

    if (frame_idx > 0 && (frame_idx % 600 == 599)) {
      LOG(INFO) << "[VideoExportor] Frame=" << frame_idx
                << " used_sws=" << (sws_to_out ? 1 : 0)
                << " dec_fmt=" << av_get_pix_fmt_name(dec_fmt)
                << " out_fmt=" << av_get_pix_fmt_name(out_fmt);
    }

    ++frame_idx;
    av_frame_unref(dec_frame);
  }

  if (sws_to_out) sws_freeContext(sws_to_out);
  if (src_conv) av_frame_free(&src_conv);
  av_frame_free(&canvas);
  writer.Close();

  // ---- Remux audio back (smart: copy AAC, transcode others to AAC) ----
  if (!stop_) {
    std::string remux_err;
    bool remux_ok = RemuxCopyAudioFromSrc(src_video_path, tmp_video_only, final_out, &remux_err);

    if (!remux_ok) {
      LOG(WARNING) << "[VideoExportor] Remux audio failed: " << remux_err
                   << " (fallback to video-only output)";

      std::string mv_err;
      if (!ReplaceFileByRenameOrCopy(tmp_video_only, final_out, &mv_err)) {
        LOG(ERROR) << "[VideoExportor] Failed to publish video-only file: " << mv_err
                   << " tmp=" << tmp_video_only << " final=" << final_out;
      } else {
        std::string vfy_err;
        if (!VerifyMediaReadable(final_out, &vfy_err)) {
          LOG(ERROR) << "[VideoExportor] Published video-only is not readable: " << vfy_err
                     << " final=" << final_out;
        }
      }
    } else {
      std::remove(tmp_video_only.c_str());
    }

    {
      std::lock_guard<std::mutex> lk(cb_mtx_);
      for (const auto& cb : done_cbs_) {
        if (cb) cb();
      }
    }
  } else {
    LOG(INFO) << "[VideoExportor] Export stopped; keep tmp: " << tmp_video_only;
  }

  running_ = false;
  LOG(INFO) << "[VideoExportor] Worker exit. frames=" << frame_idx
            << (stop_ ? " (stopped)" : "");
}

}  // namespace airsteady
