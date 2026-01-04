#include "yolo/postprocess.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "yolo/nms.h"

namespace airsteady {
namespace yolo {

static inline void SetErr(std::string* err, const std::string& msg) {
  if (err) *err = msg;
}

static inline float Sigmoid(float x) {
  // stable-ish sigmoid
  if (x >= 0) {
    const float z = std::exp(-x);
    return 1.0f / (1.0f + z);
  } else {
    const float z = std::exp(x);
    return z / (1.0f + z);
  }
}

static inline float Clamp(float v, float lo, float hi) {
  return std::max(lo, std::min(hi, v));
}

struct ParsedDet {
  RawDet det;
  std::vector<float> mask_coeff;  // len = mask_dim
};

static bool ParseDetTensor(const OrtOutput& det_out,
                           int* out_n,
                           int* out_c,
                           bool* out_is_c_first,
                           std::string* err) {
  // Accept:
  //  - [1,C,N]
  //  - [1,N,C]
  if (det_out.shape.size() != 3) {
    SetErr(err, "det_out shape must be 3D [1,C,N] or [1,N,C].");
    return false;
  }
  const int64_t d0 = det_out.shape[0];
  const int64_t d1 = det_out.shape[1];
  const int64_t d2 = det_out.shape[2];
  if (d0 != 1) {
    SetErr(err, "det_out first dim must be 1.");
    return false;
  }
  // Heuristic: C is usually <= 256; N is usually large (8400).
  bool c_first = (d1 <= 512 && d2 >= d1);
  int C = static_cast<int>(c_first ? d1 : d2);
  int N = static_cast<int>(c_first ? d2 : d1);

  *out_n = N;
  *out_c = C;
  *out_is_c_first = c_first;
  return true;
}

static bool ParseProtoTensor(const OrtOutput& proto_out,
                             int* mask_dim,
                             int* mh,
                             int* mw,
                             std::string* err) {
  // Accept [1,mask_dim,mh,mw]
  if (proto_out.shape.size() != 4 || proto_out.shape[0] != 1) {
    SetErr(err, "proto_out shape must be [1,mask_dim,mh,mw].");
    return false;
  }
  const int md = static_cast<int>(proto_out.shape[1]);
  const int h = static_cast<int>(proto_out.shape[2]);
  const int w = static_cast<int>(proto_out.shape[3]);
  if (md <= 0 || h <= 0 || w <= 0) {
    SetErr(err, "proto_out has invalid dims.");
    return false;
  }
  *mask_dim = md;
  *mh = h;
  *mw = w;
  return true;
}

bool PostprocessYoloSeg(const YoloConfig& cfg,
                        const cv::Size& src_size,
                        const LetterBoxInfo& lb,
                        const OrtOutput& det_out,
                        const OrtOutput& proto_out,
                        YoloResult* out,
                        std::string* err) {
  if (!out) {
    SetErr(err, "out is null.");
    return false;
  }
  out->dets.clear();

  int N = 0, C = 0;
  bool c_first = false;
  if (!ParseDetTensor(det_out, &N, &C, &c_first, err)) return false;

  int mask_dim = 0, mh = 0, mw = 0;
  if (!ParseProtoTensor(proto_out, &mask_dim, &mh, &mw, err)) return false;

  // We assume C = 4 + nc + mask_dim.
  const int nc = C - 4 - mask_dim;
  if (nc <= 0) {
    SetErr(err, "Invalid det_out channels: expected C = 4 + nc + mask_dim.");
    return false;
  }

  auto at = [&](int n, int c) -> float {
    // det_out.data layout:
    // if c_first: [1,C,N] => index = c*N + n
    // else:       [1,N,C] => index = n*C + c
    const size_t idx = c_first
        ? static_cast<size_t>(c) * static_cast<size_t>(N) + static_cast<size_t>(n)
        : static_cast<size_t>(n) * static_cast<size_t>(C) + static_cast<size_t>(c);
    return det_out.data[idx];
  };

  std::vector<ParsedDet> parsed;
  parsed.reserve(static_cast<size_t>(N));

  for (int i = 0; i < N; ++i) {
    // Ultralytics export: first 4 are [cx,cy,w,h] in input-space pixels.
    const float cx = at(i, 0);
    const float cy = at(i, 1);
    const float w  = at(i, 2);
    const float h  = at(i, 3);

    // class scores: next nc
    int best_cls = -1;
    float best_score = -1.0f;
    for (int c = 0; c < nc; ++c) {
      const float s = at(i, 4 + c);
      if (s > best_score) {
        best_score = s;
        best_cls = c;
      }
    }

    if (best_score < cfg.conf_thresh) continue;

    ParsedDet pd;
    pd.det.class_id = best_cls;
    pd.det.score = best_score;

    float x1 = cx - w * 0.5f;
    float y1 = cy - h * 0.5f;
    float x2 = cx + w * 0.5f;
    float y2 = cy + h * 0.5f;

    // Undo letterbox: (x - pad)/scale
    x1 = (x1 - static_cast<float>(lb.pad_x)) / lb.scale;
    y1 = (y1 - static_cast<float>(lb.pad_y)) / lb.scale;
    x2 = (x2 - static_cast<float>(lb.pad_x)) / lb.scale;
    y2 = (y2 - static_cast<float>(lb.pad_y)) / lb.scale;

    x1 = Clamp(x1, 0.0f, static_cast<float>(src_size.width - 1));
    y1 = Clamp(y1, 0.0f, static_cast<float>(src_size.height - 1));
    x2 = Clamp(x2, 0.0f, static_cast<float>(src_size.width - 1));
    y2 = Clamp(y2, 0.0f, static_cast<float>(src_size.height - 1));

    pd.det.x1 = x1;
    pd.det.y1 = y1;
    pd.det.x2 = x2;
    pd.det.y2 = y2;

    pd.mask_coeff.resize(static_cast<size_t>(mask_dim));
    for (int k = 0; k < mask_dim; ++k) {
      pd.mask_coeff[static_cast<size_t>(k)] = at(i, 4 + nc + k);
    }

    parsed.push_back(std::move(pd));
  }

  if (parsed.empty()) return true;

  // NMS
  std::vector<RawDet> raw;
  raw.reserve(parsed.size());
  for (int i = 0; i < static_cast<int>(parsed.size()); ++i) {
    RawDet d = parsed[i].det;
    d.keep_index = i;
    raw.push_back(d);
  }

  const auto keep = NmsIndices(raw, cfg.iou_thresh, cfg.class_agnostic_nms, cfg.top_k);

  // Prepare proto matrix: [mask_dim, mh*mw]
  // proto_out.data is [1,mask_dim,mh,mw] contiguous.
  const int HW = mh * mw;
  cv::Mat proto_mat(mask_dim, HW, CV_32F);
  {
    // copy per channel
    for (int c = 0; c < mask_dim; ++c) {
      const float* src = proto_out.data.data() + static_cast<size_t>(c) * static_cast<size_t>(HW);
      float* dst = proto_mat.ptr<float>(c);
      std::memcpy(dst, src, static_cast<size_t>(HW) * sizeof(float));
    }
  }

  out->dets.reserve(keep.size());

  for (int kidx : keep) {
    const int pi = raw[kidx].keep_index;
    const ParsedDet& pd = parsed[pi];

    Det d;
    d.class_id = pd.det.class_id;
    d.score = pd.det.score;

    const int x1i = static_cast<int>(std::floor(pd.det.x1));
    const int y1i = static_cast<int>(std::floor(pd.det.y1));
    const int x2i = static_cast<int>(std::ceil(pd.det.x2));
    const int y2i = static_cast<int>(std::ceil(pd.det.y2));

    const int bw = std::max(0, x2i - x1i);
    const int bh = std::max(0, y2i - y1i);
    d.box = cv::Rect(x1i, y1i, bw, bh) &
            cv::Rect(0, 0, src_size.width, src_size.height);

    if (cfg.return_masks) {
      // mask logits in proto space: [1, HW] = coeff(1,mask_dim) * proto(mask_dim,HW)
      cv::Mat coeff(1, mask_dim, CV_32F);
      for (int c = 0; c < mask_dim; ++c) coeff.at<float>(0, c) = pd.mask_coeff[static_cast<size_t>(c)];

      cv::Mat mask_flat;  // 1 x HW
      cv::gemm(coeff, proto_mat, 1.0, cv::Mat(), 0.0, mask_flat);

      // reshape to mh x mw
      cv::Mat mask_small(mh, mw, CV_32F, mask_flat.ptr<float>(0));

      // sigmoid + resize to input size
      cv::Mat mask_sig(mh, mw, CV_32F);
      for (int y = 0; y < mh; ++y) {
        const float* src = mask_small.ptr<float>(y);
        float* dst = mask_sig.ptr<float>(y);
        for (int x = 0; x < mw; ++x) dst[x] = Sigmoid(src[x]);
      }

      cv::Mat mask_in;
      cv::resize(mask_sig, mask_in, cv::Size(lb.in_w, lb.in_h), 0, 0, cv::INTER_LINEAR);

      // remove padding to get resized region (new_w,new_h), then resize to original size
      const int rx = lb.pad_x;
      const int ry = lb.pad_y;
      const int rw = lb.resized_w;
      const int rh = lb.resized_h;

      cv::Rect roi(rx, ry, rw, rh);
      roi &= cv::Rect(0, 0, mask_in.cols, mask_in.rows);

      cv::Mat mask_crop = mask_in(roi);

      cv::Mat mask_orig_f;
      cv::resize(mask_crop, mask_orig_f, src_size, 0, 0, cv::INTER_LINEAR);

      // binarize
      cv::Mat mask_u8(src_size, CV_8UC1);
      for (int y = 0; y < src_size.height; ++y) {
        const float* src = mask_orig_f.ptr<float>(y);
        uint8_t* dst = mask_u8.ptr<uint8_t>(y);
        for (int x = 0; x < src_size.width; ++x) {
          dst[x] = (src[x] >= cfg.mask_thresh) ? 255 : 0;
        }
      }

      // optional: crop to bbox to reduce storage
      if (d.box.area() > 0) {
        d.mask = mask_u8(d.box).clone();
      } else {
        d.mask = cv::Mat();
      }
    }

    out->dets.push_back(std::move(d));
  }

  return true;
}

}  // namespace yolo
}  // namespace airsteady
