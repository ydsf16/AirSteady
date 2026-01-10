#pragma once

#include <vector>

namespace airsteady {
namespace yolo {

struct RawDet {
  float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
  float score = 0;
  int class_id = -1;
  int keep_index = -1;  // optional index mapping
};

float IoU(const RawDet& a, const RawDet& b);

std::vector<int> NmsIndices(const std::vector<RawDet>& dets,
                            float iou_thresh,
                            bool class_agnostic,
                            int top_k);

}  // namespace yolo
}  // namespace airsteady
