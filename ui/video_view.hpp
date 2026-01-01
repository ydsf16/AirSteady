#pragma once

#include <QWidget>
#include <QPixmap>
#include <QRectF>
#include <QColor>

namespace airsteady {

class VideoView : public QWidget {
  Q_OBJECT
 public:
  explicit VideoView(const QString& title, QWidget* parent = nullptr);

  // 设置要显示的视频帧
  void setFramePixmap(const QPixmap& pixmap);
  void clearFrame();

  // 主 overlay 文本（提示/说明）
  void setOverlay(const QString& text,
                  bool center,
                  const QColor& color = QColor());
  void clearOverlay();

  // 警告 overlay（例如裁切超范围）
  void setWarnOverlay(const QString& text,
                      const QColor& color = QColor());
  void clearWarnOverlay();

  // 目标框（0~1 归一化坐标，可留作扩展）
  void setTargetRectNorm(const QRectF& rect_norm);
  void clearTargetRect();

  // 控制是否显示中间的大播放按钮
  void setPlayIconVisible(bool visible);

 signals:
  // 点击视频内容区域时发出（归一化坐标）
  void clicked(double x_norm, double y_norm);

 protected:
  void paintEvent(QPaintEvent* event) override;
  void mousePressEvent(QMouseEvent* event) override;

 private:
  QString title_;
  QPixmap pixmap_;
  bool has_frame_ = false;

  QString overlay_text_;
  bool has_overlay_ = false;
  QColor overlay_color_;
  bool overlay_center_ = true;

  QString warn_overlay_text_;
  bool has_warn_overlay_ = false;
  QColor warn_overlay_color_;

  QColor overlay_bg_{0, 0, 0, 120};

  QRectF target_rect_norm_;  // 0~1
  bool has_target_rect_ = false;

  QRectF last_draw_rect_;  // 实际图像绘制区域

  bool show_play_icon_ = false;
};

}  // namespace airsteady
