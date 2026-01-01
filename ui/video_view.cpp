#include "video_view.hpp"

#include <QPainter>
#include <QMouseEvent>
#include <QFont>
#include <QFontMetrics>
#include <QRadialGradient>
#include <QPen>
#include <QPolygonF>

namespace airsteady {

VideoView::VideoView(const QString& title, QWidget* parent)
    : QWidget(parent), title_(title) {
  setMinimumSize(320, 240);
}

void VideoView::setFramePixmap(const QPixmap& pixmap) {
  pixmap_ = pixmap;
  has_frame_ = !pixmap_.isNull();
  update();
}

void VideoView::clearFrame() {
  pixmap_ = QPixmap();
  has_frame_ = false;
  last_draw_rect_ = QRectF();
  update();
}

void VideoView::setOverlay(const QString& text,
                           bool center,
                           const QColor& color) {
  overlay_text_ = text;
  has_overlay_ = !overlay_text_.isEmpty();
  overlay_center_ = center;
  if (color.isValid()) {
    overlay_color_ = color;
  } else {
    overlay_color_ = QColor(255, 255, 255);
  }
  update();
}

void VideoView::clearOverlay() {
  overlay_text_.clear();
  has_overlay_ = false;
  update();
}

void VideoView::setWarnOverlay(const QString& text,
                               const QColor& color) {
  warn_overlay_text_ = text;
  has_warn_overlay_ = !warn_overlay_text_.isEmpty();
  if (color.isValid()) {
    warn_overlay_color_ = color;
  } else {
    warn_overlay_color_ = QColor(255, 200, 200);
  }
  update();
}

void VideoView::clearWarnOverlay() {
  warn_overlay_text_.clear();
  has_warn_overlay_ = false;
  update();
}

void VideoView::setTargetRectNorm(const QRectF& rect_norm) {
  target_rect_norm_ = rect_norm;
  has_target_rect_ = true;
  update();
}

void VideoView::clearTargetRect() {
  has_target_rect_ = false;
  update();
}

void VideoView::setPlayIconVisible(bool visible) {
  show_play_icon_ = visible;
  update();
}

void VideoView::paintEvent(QPaintEvent* /*event*/) {
  QPainter painter(this);
  painter.fillRect(rect(), QColor(32, 32, 32));

  // 标题
  painter.setPen(QColor(220, 220, 220));
  QFont font = painter.font();
  font.setBold(true);
  painter.setFont(font);
  painter.drawText(10, 20, title_);

  // 内容区域
  QRect content_rect = rect().adjusted(5, 25, -5, -5);

  last_draw_rect_ = QRectF();

  if (has_frame_ && !pixmap_.isNull() && !content_rect.isEmpty()) {
    QPixmap fit = pixmap_.scaled(content_rect.size(),
                                 Qt::KeepAspectRatio,
                                 Qt::SmoothTransformation);

    int x = content_rect.x() + (content_rect.width() - fit.width()) / 2;
    int y = content_rect.y() + (content_rect.height() - fit.height()) / 2;

    painter.save();
    painter.setClipRect(content_rect);
    painter.drawPixmap(x, y, fit);
    painter.restore();

    last_draw_rect_ = QRectF(x, y, fit.width(), fit.height());
  }

  // 目标框（如有）
  if (has_target_rect_ && !last_draw_rect_.isNull()) {
    const QRectF& r = target_rect_norm_;
    double rx = r.x();
    double ry = r.y();
    double rw = r.width();
    double rh = r.height();

    double x = last_draw_rect_.x() + rx * last_draw_rect_.width();
    double y = last_draw_rect_.y() + ry * last_draw_rect_.height();
    double w = rw * last_draw_rect_.width();
    double h = rh * last_draw_rect_.height();

    painter.setPen(QColor(0, 255, 0));
    painter.drawRect(QRectF(x, y, w, h));
  }

  // overlay 文本
  if (has_overlay_ && !overlay_text_.isEmpty()) {
    QFont ofont("Microsoft YaHei", 11);
    painter.setFont(ofont);
    QFontMetrics metrics(ofont);

    int text_width = metrics.horizontalAdvance(overlay_text_);
    int text_height = metrics.height();

    double tx = content_rect.center().x() - text_width / 2.0;
    double ty = overlay_center_
                    ? content_rect.center().y() - text_height / 2.0
                    : content_rect.y() + content_rect.height() * 0.1;

    QRectF bg_rect(tx - 10,
                   ty - text_height,
                   text_width + 20,
                   text_height + 40);

    painter.fillRect(bg_rect, overlay_bg_);
    painter.setPen(overlay_color_);
    painter.drawText(bg_rect,
                     Qt::AlignCenter | Qt::AlignVCenter,
                     overlay_text_);
  }

  // 警告 overlay
  if (has_warn_overlay_ && !warn_overlay_text_.isEmpty()) {
    QFont wfont("Microsoft YaHei", 11);
    painter.setFont(wfont);
    QFontMetrics metrics(wfont);

    int text_width = metrics.horizontalAdvance(warn_overlay_text_);
    int text_height = metrics.height();

    double tx = content_rect.center().x() - text_width / 2.0;
    double ty = content_rect.center().y() - text_height / 2.0;

    QRectF bg_rect(tx - 10,
                   ty - text_height,
                   text_width + 20,
                   text_height + 40);

    painter.fillRect(bg_rect, overlay_bg_);
    painter.setPen(warn_overlay_color_);
    painter.drawText(bg_rect,
                     Qt::AlignCenter | Qt::AlignVCenter,
                     warn_overlay_text_);
  }

  // 中间的大播放按钮
  if (show_play_icon_ && !content_rect.isEmpty()) {
    painter.setRenderHint(QPainter::Antialiasing, true);

    double size =
        std::min(content_rect.width(), content_rect.height()) * 0.10;
    double radius = size / 2.0;
    QPointF center = content_rect.center();

    // 阴影
    QRectF shadow_rect(center.x() - radius - 3,
                       center.y() - radius - 3,
                       size + 6,
                       size + 6);
    painter.setBrush(QColor(0, 0, 0, 120));
    painter.setPen(Qt::NoPen);
    painter.drawEllipse(shadow_rect);

    // 渐变圆
    QRectF circle_rect(center.x() - radius,
                       center.y() - radius,
                       size,
                       size);
    QRadialGradient grad(circle_rect.center(), radius);
    grad.setColorAt(0.0, QColor(255, 255, 255, 40));
    grad.setColorAt(1.0, QColor(0, 0, 0, 190));
    painter.setBrush(grad);
    painter.setPen(QPen(QColor(255, 255, 255, 220), 1.6));
    painter.drawEllipse(circle_rect);

    // ▶ 三角形
    double tri_r = radius * 0.58;
    QPointF p1(center.x() - tri_r * 0.35, center.y() - tri_r);
    QPointF p2(center.x() - tri_r * 0.35, center.y() + tri_r);
    QPointF p3(center.x() + tri_r, center.y());
    QPolygonF triangle;
    triangle << p1 << p2 << p3;

    painter.setBrush(QColor(255, 255, 255, 235));
    painter.setPen(Qt::NoPen);
    painter.drawPolygon(triangle);
  }
}

void VideoView::mousePressEvent(QMouseEvent* event) {
  if (event->button() == Qt::LeftButton && !last_draw_rect_.isNull()) {
    QPointF pos = event->position();
    if (!last_draw_rect_.contains(pos)) {
      QWidget::mousePressEvent(event);
      return;
    }

    double x_norm =
        (pos.x() - last_draw_rect_.x()) / last_draw_rect_.width();
    double y_norm =
        (pos.y() - last_draw_rect_.y()) / last_draw_rect_.height();
    x_norm = std::clamp(x_norm, 0.0, 1.0);
    y_norm = std::clamp(y_norm, 0.0, 1.0);

    emit clicked(x_norm, y_norm);
    event->accept();
    return;
  }
  QWidget::mousePressEvent(event);
}

}  // namespace airsteady
