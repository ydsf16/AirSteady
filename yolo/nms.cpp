#include "yolo/nms.h"

#include <algorithm>
#include <cmath>

namespace airsteady {
namespace yolo {

static inline float Area(const RawDet& d) {
  const float w = std::max(0.0f, d.x2 - d.x1);
  const float h = std::max(0.0f, d.y2 - d.y1);
  return w * h;
}

float IoU(const RawDet& a, const RawDet& b) {
  const float xx1 = std::max(a.x1, b.x1);
  const float yy1 = std::max(a.y1, b.y1);
  const float xx2 = std::min(a.x2, b.x2);
  const float yy2 = std::min(a.y2, b.y2);
  const float w = std::max(0.0f, xx2 - xx1);
  const float h = std::max(0.0f, yy2 - yy1);
  const float inter = w * h;
  const float uni = Area(a) + Area(b) - inter;
  return (uni <= 0.0f) ? 0.0f : (inter / uni);
}

std::vector<int> NmsIndices(const std::vector<RawDet>& dets,
                            float iou_thresh,
                            bool class_agnostic,
                            int top_k) {
  std::vector<int> idx(dets.size());
  for (int i = 0; i < static_cast<int>(idx.size()); ++i) idx[i] = i;

  std::sort(idx.begin(), idx.end(), [&](int i, int j) {
    return dets[i].score > dets[j].score;
  });

  std::vector<int> keep;
  keep.reserve(std::min(top_k, static_cast<int>(dets.size())));

  for (int ii = 0; ii < static_cast<int>(idx.size()); ++ii) {
    const int i = idx[ii];
    bool suppressed = false;
    for (int k : keep) {
      if (!class_agnostic && dets[i].class_id != dets[k].class_id) continue;
      if (IoU(dets[i], dets[k]) > iou_thresh) { suppressed = true; break; }
    }
    if (!suppressed) {
      keep.push_back(i);
      if (static_cast<int>(keep.size()) >= top_k) break;
    }
  }
  return keep;
}

}  // namespace yolo
}  // namespace airsteady
