#pragma once

#include <string>

namespace airsteady {

struct StableParams {
    std::string track_type = "air_plane";

    double smooth_ratio = 1.0; // 0.0 - 1.0;

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

}  // namespace airsteady