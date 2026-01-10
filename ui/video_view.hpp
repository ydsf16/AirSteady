#pragma once

#include <QWidget>
#include <QColor>
#include <QImage>
#include <QRectF>
#include <QString>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <vector>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "common/types.h"

namespace airsteady {

class VideoView : public QWidget {
  Q_OBJECT
 public:
  explicit VideoView(const QString& title, QWidget* parent = nullptr);
  ~VideoView() override = default;

  // Tracking stage: draw proxy_bgr + seg overlay (bbox/points/timing).
  void SetTrackFrame(const FrameTrackingResultPreview& track_res);

  // Preview stage:
  // - left=true  : draw tracking view (proxy_bgr + seg overlay from frame_preview.track_res.seg_detect_res)
  // - left=false : draw stabilized view (proxy_bgr shifted by global_offset + stable_res.delta)
  void SetPreviewFrame(const FramePreview& frame_preview,
                       const Eigen::Vector2d& global_offset,
                       bool left = false);

  void ClearFrame();

  void SetOverlay(const QString& text,
                  bool center,
                  const QColor& color = QColor(255, 255, 255));
  void ClearOverlay();

  void SetWarnOverlay(const QString& text,
                      const QColor& color = QColor(255, 180, 0));
  void ClearWarnOverlay();

  // 导出 / 长任务进度条接口：ratio ∈ [0,1]
  void UpdateProgressBar(double ratio);
  void ClearProgressBar();

 signals:
  // Optional: if you later want click-to-select ROI, emit normalized rect.
  void TargetRectSelected(const QRectF& rect_norm);  // 0~1 in image coords

 protected:
  void paintEvent(QPaintEvent* event) override;
  void mousePressEvent(QMouseEvent* event) override;

 private:
  enum class RenderMode {
    kNone = 0,
    kTracking,
    kPreviewLeft,
    kPreviewRight,
  };

  struct DrawDet {
    QRectF box;      // in image pixel coords
    int class_id = -1;
    float score = 0.0f;
    bool selected = false;
  };

  struct DrawOverlay {
    int frame_idx = -1;
    std::int64_t time_ns = 0;

    std::vector<DrawDet> dets;
    bool has_selected = false;
    QRectF selected_box;  // pixel coords

    std::vector<QPointF> good_pts;  // pixel coords

    SegDetectorTiming timing;
  };

 private:
  static QImage CvMatToQImageDeepCopy(const cv::Mat& mat_bgr_or_gray);
  static DrawOverlay BuildOverlayFromSegRes(const SegDetectorRes& seg);

  void RequestRepaintAsync();
  QRectF ComputeLetterboxRect(int img_w, int img_h) const;
  QPointF MapImageToWidget(const QPointF& p_img, const QRectF& draw_rect,
                           int img_w, int img_h) const;
  QRectF MapImageRectToWidget(const QRectF& r_img, const QRectF& draw_rect,
                              int img_w, int img_h) const;

  void PaintTitleAndOverlays(QPainter* painter, const QRectF& draw_rect);
  void PaintTrackingOverlay(QPainter* painter, const QRectF& draw_rect,
                            int img_w, int img_h, const DrawOverlay& ov);
  void PaintStabilizedView(QPainter* painter, const QRectF& draw_rect,
                           const QImage& img, const Eigen::Vector2d& shift_px);

 private:
  QString title_;

  mutable std::mutex mu_;

  RenderMode mode_ = RenderMode::kNone;

  // Frame image stored as deep-copied QImage (no dependency on cv::Mat lifetime).
  QImage img_;
  bool has_frame_ = false;

  // Tracking/left-preview overlay
  std::optional<DrawOverlay> overlay_;

  // Right-preview offsets
  Eigen::Vector2d global_offset_px_ = Eigen::Vector2d::Zero();
  FrameStableResult stable_res_;
  bool has_stable_ = false;

  // UI overlay strings
  QString overlay_text_;
  bool has_overlay_ = false;
  QColor overlay_color_{255, 255, 255};
  bool overlay_center_ = true;

  QString warn_overlay_text_;
  bool has_warn_overlay_ = false;
  QColor warn_overlay_color_{255, 180, 0};

  QColor overlay_bg_{0, 0, 0, 140};

  // (Optional) selection rectangle (0~1 normalized in image coords)
  QRectF target_rect_norm_;
  bool has_target_rect_ = false;

  // For click handling / debugging
  QRectF last_draw_rect_;

  // Progress bar state.
  bool has_progress_bar_ = false;
  double progress_ratio_ = 0.0;  // [0,1]
};

}  // namespace airsteady
